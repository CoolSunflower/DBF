import os
import time
import torch
import torch.nn as nn
import numpy as np
import datasets
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
        # attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print("ms", model.config.max_position_embeddings)
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev, n_samples=256):
    print('Starting ...')
    
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev) 
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
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
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    comp_inps = inps.clone()
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings']

    print('Ready.')

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        for j in range(n_samples):
            inps[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]

        imp = layer.mlp.down_proj.o_norm
        
        for name in [
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.k_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ]:
            to_opt = {n: p for n, p in layer.named_parameters() if "weight" in n and "layernorm" not in n}
            if len(to_opt) > 0 and (("q_proj" in name and i >= 1) or "k_proj" in name):
                
                for n, p in to_opt.items():
                    p.requires_grad = True
                
                # Using standard AdamW instead of fused adam for compatibility
                lr = 3e-5
                opt = torch.optim.AdamW(to_opt.values(), lr=lr, weight_decay=1e-4)
                sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=n_samples*4 // 8, cycle_momentum=False)
                
                err_before = 0
                for j in range(n_samples):
                    cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                    err_before += ((cur_out.float() - inps[j].cuda().float()).square() * imp).mean().item()

                print(f"Layer {i} {name} err before: {err_before}")

                with torch.enable_grad():
                    for ep in range(4):
                        err_total = 0
                        for j in range(n_samples):
                            cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
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
                    cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                    err_after += ((cur_out.float() - inps[j].cuda().float()).square() * imp).mean().item()
                print(f"Layer {i} {name} err after: {err_after}")
                        
            print(i, name)
            print('Pruning ...')
            lx = subset[name]
            
            s1 = time.time()
            W2, Ac, Bb, mid, = factorize(lx)
            W2 = W2.T
            err_spxsp = (W2.T - lx.weight).square().sum().item()
            print("err_spxsp", err_spxsp)
            
            # Log bit allocation
            if not hasattr(model, 'layer_bits'):
                model.layer_bits = {}
            model.layer_bits[f"layer_{i}.{name}"] = mid
            
            lx.weight.data = W2.T.to(lx.weight)
            lx.weight.A = Ac
            lx.weight.B = Bb
            lx.weight.mid = mid
            parts = name.split('.')
            block = getattr(layer, parts[0])
            setattr(block, parts[1], replace(lx))
            
        for j in range(n_samples):
            comp_inps[j] = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        
    # Save bit allocation
    with open("bit_allocation_tinyllama.json", "w") as f:
        json.dump(model.layer_bits, f, indent=4)

import json

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    print(f"Loading model: {model_name}")
    
    model = get_opt(model_name)
    model.eval()
    
    # Calibration data
    n_samples = 256
    print("Loading calibration data...")
    dataloader, _ = get_loaders("wikitext2", nsamples=n_samples, seed=0, seqlen=model.seqlen, model=model_name)
    
    # Convert list of tensors to a single tensor if needed, or handle in loop
    # The original code expects a list of (input, target) tuples or similar from get_loaders
    # But get_wikitext2 returns a list of (inp, tar). We just need inp.
    dataloader = [x[0] for x in dataloader]
    dataloader = torch.cat(dataloader, dim=0) # (n_samples, seqlen)
    
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

    # Run forward/backward for norms
    # We need a loss function to backprop
    # Using a simple output matching or just running data through
    # The original code used linear_cross_entropy. We'll use standard CE.
    
    for idx, bx in enumerate(dataloader):
        bx = bx.cuda().unsqueeze(0) # (1, seqlen)
        
        # Forward
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
            # Store the mean sensitivity (scalar) for the plot
            layer_sensitivity[n] = m.o_norm.mean().item()
            
    # Save sensitivity
    with open("sensitivity_tinyllama.json", "w") as f:
        json.dump(layer_sensitivity, f, indent=4)
    
    # Run compression
    start = time.time()
    model.cpu()
    opt_sequential(model, dataloader, DEV, n_samples=n_samples)
    print("Total compression time:", time.time() - start)
    
    # Save model
    save_path = "tinyllama_dbf_compressed.pt"
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
