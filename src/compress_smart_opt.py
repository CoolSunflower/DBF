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
from src.compress_opt import get_opt, opt_sequential

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

DEV = torch.device('cuda:0')

def calculate_smart_mid(sensitivity_file, model):
    print("Calculating Smart Mid values...")
    with open(sensitivity_file, "r") as f:
        sensitivity = json.load(f)
    
    # Sort layers by sensitivity
    sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1])
    n_layers = len(sorted_layers)
    
    # Define bins (Simple heuristic: Bottom 30% -> Low Rank, Top 30% -> High Rank, Middle -> Mid Rank)
    # Base rank (Uniform) is usually around 1.5 bits equivalent.
    # Let's say:
    # Low: 1.0 bits
    # Mid: 1.5 bits
    # High: 2.0 bits
    
    smart_mid_map = {}
    
    for i, (name, score) in enumerate(sorted_layers):
        # Find the module to get dimensions
        # name is like "model.decoder.layers.0.fc1"
        # We need to find the module in the model
        try:
            module = model.get_submodule(name)
            in_f = module.in_features
            out_f = module.out_features
            
            # Calculate mid for different bit targets
            # Total bits = (in*mid + mid*out + (in+mid+out)*16)
            # Bits per weight = Total bits / (in*out)
            # Approx: mid * (in+out) / (in*out) = bits
            # mid = bits * (in*out) / (in+out)
            
            percentile = i / n_layers
            if percentile < 0.3:
                target_bits = 1.0
            elif percentile < 0.7:
                target_bits = 1.5
            else:
                target_bits = 2.0
                
            mid = int(target_bits * (in_f * out_f) / (in_f + out_f))
            
            # Ensure mid is at least 1 and not larger than min(in, out)
            mid = max(1, min(mid, min(in_f, out_f)))
            
            # Map name to layer index format used in opt_sequential
            # name: model.decoder.layers.0.fc1
            # target: layer_0.fc1
            parts = name.split('.')
            if "layers" in parts:
                layer_idx = parts[parts.index("layers") + 1]
                sub_name = parts[-1] # fc1, fc2, k_proj etc
                if "self_attn" in parts:
                    sub_name = "self_attn." + sub_name
                
                key = f"layer_{layer_idx}.{sub_name}"
                smart_mid_map[key] = mid
                
        except AttributeError:
            print(f"Could not find module {name}")
            continue
            
    return smart_mid_map

def main():
    model_name = "Aalaa/opt-125m-finetuned-wikitext2"
    print(f"Loading model: {model_name}")
    
    model = get_opt(model_name)
    model.eval()
    
    # Calibration data
    n_samples = 128
    print("Loading calibration data...")
    dataloader, _ = get_loaders("wikitext2", nsamples=n_samples, seed=0, seqlen=model.seqlen, model=model_name)
    dataloader = [x[0] for x in dataloader]
    dataloader = torch.cat(dataloader, dim=0) 
    
    # 1. Run Calibration (or load if exists)
    sensitivity_file = "sensitivity_opt.json"
    if not os.path.exists(sensitivity_file):
        print("Sensitivity file not found. Running calibration...")
        # We can reuse the logic from compress_opt.py main, but let's just call it or copy it.
        # For simplicity, I'll assume we run compress_opt.py FIRST to generate sensitivity, 
        # OR we copy the calibration block here.
        # Let's copy the calibration block for robustness.
        
        print("Pre-computing norms...")
        model.cuda()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()

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
            loss.backward()

        for h in handles:
            h.remove()
            
        for n, m in model.named_modules():
            if hasattr(m, 'o_norm'):
                layer_sensitivity[n] = m.o_norm.mean().item()
                
        with open(sensitivity_file, "w") as f:
            json.dump(layer_sensitivity, f, indent=4)
    
    # 2. Calculate Smart Mid
    smart_mid = calculate_smart_mid(sensitivity_file, model)
    print(f"Calculated smart mid for {len(smart_mid)} layers.")
    
    # 3. Pre-compute norms (REQUIRED even if sensitivity file exists)
    print("Pre-computing norms for compression...")
    model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

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
            print(f"Norm calibration step {idx}/{n_samples}")
        loss.backward()

    for h in handles:
        h.remove()
        
    print("Norm computation done.")
    
    # 4. Run Compression with Smart Mid
    # We need to patch factorize to use our mid.
    # In compress_opt.py, we added logic to check for smart_mid.
    # We need to pass it to opt_sequential.
    
    start = time.time()
    model.cpu()
    opt_sequential(model, dataloader, DEV, n_samples=n_samples, smart_mid=smart_mid)
    print("Total compression time:", time.time() - start)
    
    # Save model
    save_path = "opt_dbf_smart_compressed.pt"
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
