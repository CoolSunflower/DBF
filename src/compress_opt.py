import os
import time
import torch
import torch.nn as nn
import numpy as np
import datasets
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modelutils import find_layers
from src.datautils import get_loaders
from src.dbf_core import factorize
from src.layers import replace

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

DEV = torch.device('cuda:0')

def get_opt(model_name):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    print("ms", model.config.max_position_embeddings)
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev, n_samples=256, smart_mid=None):
    print('Starting ...')
    
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # OPT structure: model.model.decoder.layers
    layers = model.model.decoder.layers
    
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (n_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # OPT doesn't use position_embeddings in forward args usually, but let's catch kwargs
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    torch.cuda.empty_cache()

    comp_inps = inps.clone()
    attention_mask = cache['attention_mask']

    print('Ready.')

    layers = model.model.decoder.layers

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        for j in range(n_samples):
            # OPT layer forward: (hidden_states, attention_mask=None, layer_head_mask=None, output_attentions=False, use_cache=False)
            inps[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0]

        # OPT structure:
        # self_attn.k_proj, v_proj, q_proj, out_proj
        # fc1, fc2
        
        # We need to find the output norm for the layer. 
        # In Llama it was layer.mlp.down_proj.o_norm
        # In OPT, the last linear layer is fc2.
        imp = layer.fc2.o_norm
        
        for name in [
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "self_attn.k_proj",
            "fc1",
            "fc2",
        ]:
            to_opt = {n: p for n, p in layer.named_parameters() if "weight" in n and "layernorm" not in n}
            # Filter for current sub-module if needed, but here we optimize all params in the layer 
            # relative to the reconstruction error of the layer output?
            # The original code optimized specific projections.
            
            # Let's stick to the original logic: optimize params when processing specific layers
            # But for simplicity in this port, we might skip the fine-tuning loop if it's complex to map
            # Or we can try to map it.
            
            # Fine-tuning (PV-Tuning equivalent)
            # We optimize the weights in the layer to minimize reconstruction error of the layer output
            # This is crucial for maintaining performance.
            
            # User requested to NOT skip any fine-tuning.
            # We will run it for all layers.
            
            for n, p in to_opt.items():
                p.requires_grad = True
            
            lr = 3e-5
            opt = torch.optim.AdamW(to_opt.values(), lr=lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=n_samples*4 // 8, cycle_momentum=False)
            
            err_before = 0
            for j in range(n_samples):
                cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0]
                err_before += ((cur_out.float() - inps[j].cuda().float()).square() * imp).mean().item()

            print(f"Layer {i} {name} err before: {err_before}")

            with torch.enable_grad():
                for ep in range(4):
                    err_total = 0
                    for j in range(n_samples):
                        cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0]
                        err = ((cur_out.float() - inps[j].cuda().float()).square() * imp).sum()
                        err.backward()
                        if j % 8 == 7:
                            opt.step()
                            sch.step()
                            layer.zero_grad(set_to_none=True)
                        err_total += err.item() / inps.shape[1] / inps.shape[2]
                    print(f"Epoch {ep} err: {err_total}")

            err_after = 0
            for j in range(n_samples):
                cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0]
                err_after += ((cur_out.float() - inps[j].cuda().float()).square() * imp).mean().item()
            print(f"Layer {i} {name} err after: {err_after}")
            
            print(i, name)
            print('Pruning ...')
            lx = subset[name]
            
            # Determine mid
            target_mid = None
            if smart_mid and f"layer_{i}.{name}" in smart_mid:
                target_mid = smart_mid[f"layer_{i}.{name}"]
                print(f"Using smart mid: {target_mid}")
            
            s1 = time.time()
            
            W2, Ac, Bb, mid, = factorize(lx, target_mid=target_mid)
            W2 = W2.T
            err_spxsp = (W2.T - lx.weight).square().sum().item()
            print("err_spxsp", err_spxsp)
            
            # Log bit allocation
            if not hasattr(model, 'layer_bits'):
                model.layer_bits = {}
            model.layer_bits[f"layer_{i}.{name}"] = Ac.shape[1]
            
            lx.weight.data = W2.T.to(lx.weight)
            lx.weight.A = Ac
            lx.weight.B = Bb
            lx.weight.mid = mid
            parts = name.split('.')
            if len(parts) == 1:
                setattr(layer, parts[0], replace(lx))
            else:
                block = getattr(layer, parts[0])
                setattr(block, parts[1], replace(lx))
            
        for j in range(n_samples):
            comp_inps[j] = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0]
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    
    # Save model
    save_path = "opt_dbf_compressed.pt"
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

    # Save bit allocation
    with open("bit_allocation_opt.json", "w") as f:
        json.dump(model.layer_bits, f, indent=4)

def main():
    model_name = "Aalaa/opt-125m-finetuned-wikitext2"
    print(f"Loading model: {model_name}")
    
    model = get_opt(model_name)
    model.eval()
    
    # Calibration data
    n_samples = 128 # Smaller for OPT
    print("Loading calibration data...")
    dataloader, _ = get_loaders("wikitext2", nsamples=n_samples, seed=0, seqlen=model.seqlen, model=model_name)
    
    dataloader = [x[0] for x in dataloader]
    dataloader = torch.cat(dataloader, dim=0) 
    
    print("Calibration data shape:", dataloader.shape)

    # Pre-compute norms
    print("Pre-computing norms...")
    model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    # Dictionaries to store stats
    layer_sensitivity = {}
    
    def f_hook(m, i, o):
        X = i[0].detach().float()
        X = X.reshape(-1, X.shape[-1])
        if not hasattr(m, 'i_norm'):
             m.i_norm = torch.zeros(m.weight.shape[1], device=m.weight.device)
        m.i_norm += X.square().mean(dim=0)
        
    def b_hook(m, _, go):
        X = go[0].detach().float()
        X = X.reshape(-1, X.shape[-1])
        if not hasattr(m, 'o_norm'):
            m.o_norm = torch.zeros(m.weight.shape[0], device=m.weight.device)
        m.o_norm += X.square().mean(dim=0) * 1e6

    handles = []
    for n, m in model.named_modules():
        if type(m) == nn.Linear and "lm_head" not in n:
            m.i_norm = torch.zeros(m.weight.shape[1], device=m.weight.device)
            m.o_norm = torch.zeros(m.weight.shape[0], device=m.weight.device)
            handles.append(m.register_forward_hook(f_hook))
            handles.append(m.register_full_backward_hook(b_hook))

    for idx, bx in enumerate(dataloader):
        bx = bx.cuda().unsqueeze(0)
        outputs = model(bx, labels=bx)
        loss = outputs.loss
        if idx % 10 == 0:
            print(f"Calibration step {idx}/{n_samples}, Loss: {loss.item()}")
        loss.backward()

    for h in handles:
        h.remove()
        
    print("Norm computation done.")
    
    # Collect sensitivity stats
    for n, m in model.named_modules():
        if hasattr(m, 'o_norm'):
            layer_sensitivity[n] = m.o_norm.mean().item()
            
    with open("sensitivity_opt.json", "w") as f:
        json.dump(layer_sensitivity, f, indent=4)
    
    # Run compression
    start = time.time()
    model.cpu()
    opt_sequential(model, dataloader, DEV, n_samples=n_samples)
    print("Total compression time:", time.time() - start)


if __name__ == "__main__":
    main()
