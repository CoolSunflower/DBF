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
    model.seqlen = model.config.max_position_embeddings
    return model

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    print(f"Loading model: {model_name}")
    
    model = get_opt(model_name)
    model.eval()
    
    # Calibration data
    n_samples = 128
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
            
    with open("sensitivity_tinyllama.json", "w") as f:
        json.dump(layer_sensitivity, f, indent=4)
    print("Saved sensitivity_tinyllama.json")

if __name__ == "__main__":
    main()
