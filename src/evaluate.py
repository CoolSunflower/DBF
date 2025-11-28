import torch
import torch.nn as nn
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.layers import BitLinear, Mul, my_pack, my_unpack
from src.datautils import get_loaders
import math
import sys
import os
import argparse
import time
import json
import matplotlib.pyplot as plt

def plot_sensitivity(json_file, output_file="sensitivity_plot.png"):
    if not os.path.exists(json_file):
        print(f"Sensitivity file {json_file} not found.")
        return
    
    with open(json_file, "r") as f:
        data = json.load(f)
        
    values = list(data.values())
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.title("Layer Sensitivity (Output Gradient Norm)")
    plt.xlabel("Layer Index (approx)")
    plt.ylabel("Sensitivity")
    plt.savefig(output_file)
    print(f"Saved sensitivity plot to {output_file}")

def plot_bit_allocation(json_file, output_file="bit_allocation_plot.png"):
    if not os.path.exists(json_file):
        print(f"Bit allocation file {json_file} not found.")
        return
        
    with open(json_file, "r") as f:
        data = json.load(f)
        
    values = list(data.values())
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(values)), values)
    plt.title("Layer-wise Bit Allocation (Mid Rank)")
    plt.xlabel("Layer Index")
    plt.ylabel("Mid Rank")
    plt.savefig(output_file)
    print(f"Saved bit allocation plot to {output_file}")

def reconstruct_model(model, state_dict):
    """
    Replaces Linear layers in the model with DBF layers (Sequential of Mul/BitLinear)
    based on the keys present in the state_dict.
    Returns model and a dictionary of bit allocations (mid values).
    """
    print("Reconstructing model structure from state_dict...")
    
    replacements = {}
    bit_allocation = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer is replaced in state_dict
            if f"{name}.0.w" in state_dict:
                # print(f"Detected DBF layer for: {name}")
                
                in_features = module.in_features
                out_features = module.out_features
                
                # Infer mid from the first BitLinear packed weight
                bp_key = f"{name}.1.bp"
                if bp_key not in state_dict:
                    continue
                
                bp_tensor = state_dict[bp_key]
                total_elements = bp_tensor.numel() * 8
                mid = total_elements // in_features
                
                bit_allocation[name] = mid
                
                # Construct the sequence
                m0 = Mul(torch.zeros(in_features))
                bl1 = BitLinear(torch.zeros(mid, in_features))
                m2 = Mul(torch.zeros(mid))
                bl3 = BitLinear(torch.zeros(out_features, mid))
                m4 = Mul(torch.zeros(out_features))
                
                new_layer = nn.Sequential(m0, bl1, m2, bl3, m4)
                replacements[name] = new_layer

    # Apply replacements
    for name, new_module in replacements.items():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name = name.rsplit('.', 1)[1] if '.' in name else name
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            
        setattr(parent, child_name, new_module)
        
    return model, bit_allocation

@torch.no_grad()
def eval_ppl(model, tokenizer, device="cuda"):
    print("Evaluating Perplexity on Wikitext-2...")
    model.eval()
    model.to(device)
    
    # Load Wikitext-2 test data
    testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) < 2: # Skip too short sequences
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # loss is calculated on tokens < -100, so we don't need to manually shift?
            # HuggingFace models usually shift internally for CausalLM if labels are provided.
            # But let's verify: "The loss is only calculated for the tokens with labels in [0, ..., config.vocab_size]"
            # Yes.
            
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl.item()}")
    return ppl.item()

MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "checkpoint": "output/tinyllama_dbf_compressed.pt"
    },
    "llama3": {
        "name": "meta-llama/Llama-3.2-1B",
        "checkpoint": "output/llama3_dbf_compressed.pt"
    },
    "opt": {
        "name": "Aalaa/opt-125m-finetuned-wikitext2",
        "checkpoint": "output/opt_dbf_compressed.pt"
    }
}

import time
import json

def measure_speed_and_memory(model, tokenizer, device="cuda", n_tokens=100):
    print(f"Measuring inference speed and memory (generating {n_tokens} tokens)...")
    model.eval()
    model.to(device)
    
    input_text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start_time = time.time()
    _ = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    tokens_per_sec = n_tokens / duration
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"Peak Memory: {peak_memory:.2f} MB")
    
    return tokens_per_sec, peak_memory

def get_model_size(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0
    size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"Model Checkpoint Size: {size_mb:.2f} MB")
    return size_mb

def calculate_theoretical_flops(model, model_type="dense"):
    """
    Calculates theoretical FLOPs for a single forward pass of the model.
    Assumes input sequence length of 1 for simplicity (per-token cost).
    
    model_type: "dense" or "dbf"
    """
    total_ops = 0
    total_params = 0
    
    # We iterate through modules to find Linear layers (or DBF replacements)
    # Note: This is a simplified count focusing on MatMuls which dominate compute.
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Dense Linear: y = xW + b
            # W is (out, in). x is (1, in).
            # Ops: 2 * in * out (Multiply-Add)
            in_f = module.in_features
            out_f = module.out_features
            ops = 2 * in_f * out_f
            total_ops += ops
            total_params += in_f * out_f
            
        elif isinstance(module, nn.Sequential) and len(list(module.children())) == 5:
            # Check if it's our DBF structure: Mul -> BitLinear -> Mul -> BitLinear -> Mul
            children = list(module.children())
            if isinstance(children[1], BitLinear) and isinstance(children[3], BitLinear):
                # DBF Layer
                # Structure:
                # 1. Mul(v1): x * v1. Shape (in). Ops: in
                # 2. BitLinear(b1): x @ b1.T. Shape (in) @ (in, mid) -> (mid).
                #    Binary Ops: 2 * in * mid (XNOR-Popcount)
                #    But usually we weight Binary Ops less. Let's count them separately or scale them.
                #    Standard practice: 1 Binary Op = 1/16 FP Op (or 1/32). Let's use 1/16.
                # 3. Mul(mid): x * mid. Shape (mid). Ops: mid
                # 4. BitLinear(b2): x @ b2.T. Shape (mid) @ (mid, out) -> (out).
                #    Binary Ops: 2 * mid * out
                # 5. Mul(u2): x * u2. Shape (out). Ops: out
                
                # Extract shapes
                # b1 shape is in children[1].shape (mid, in)
                mid, in_f = children[1].shape
                out_f, mid2 = children[3].shape
                
                # FP Ops (Scaling)
                fp_ops = in_f + mid + out_f
                
                # Binary Ops
                bin_ops = 2 * in_f * mid + 2 * mid * out_f
                
                # Total Equivalent FP Ops
                # We assume 1 Binary Op ~= 1/16 FP Op (conservative estimate for modern GPUs)
                scaling_factor = 1/16
                total_ops += fp_ops + (bin_ops * scaling_factor)
                
                # Params (Effective bits / 16 to compare with FP16 params)
                # We store packed bits, so we count bits.
                # b1: in * mid bits
                # b2: mid * out bits
                # scales: (in + mid + out) * 16 bits
                total_bits = (in_f * mid) + (mid * out_f) + (in_f + mid + out_f) * 16
                total_params += total_bits / 16 # Equivalent FP16 params count
                
    return total_ops, total_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tinyllama', choices=MODELS.keys(), help='Model to evaluate')
    args = parser.parse_args()
    
    config = MODELS[args.model]
    model_name = config["name"]
    checkpoint_path = config["checkpoint"]
    
    results = {}
    
    print(f"=== Evaluating {args.model} ===")
    print(f"Base Model: {model_name}")
    
    # 1. Evaluate Base Model
    print("\n--- Base Model Evaluation ---")
    # Load base model on GPU if possible for speed
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto" 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    base_flops, base_params = calculate_theoretical_flops(model, "dense")
    print(f"Base Model FLOPs/token: {base_flops/1e9:.4f} GFLOPs")
    
    base_ppl = eval_ppl(model, tokenizer)
    base_speed, base_mem = measure_speed_and_memory(model, tokenizer)
    
    results['base'] = {
        'ppl': base_ppl,
        'speed': base_speed,
        'memory': base_mem,
        'flops': base_flops
    }
    
    print(f"Base Model PPL: {base_ppl:.4f}")
    
    del model
    torch.cuda.empty_cache()
    
    # 2. Evaluate Compressed Model
    print("\n--- Compressed Model Evaluation ---")
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Skipping compressed evaluation.")
        return

    # Load on CPU first for reconstruction to avoid OOM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu" 
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model, bit_allocation = reconstruct_model(model, state_dict)
    
    # Save recovered bit allocation
    with open(f"bit_allocation_{args.model}.json", "w") as f:
        json.dump(bit_allocation, f, indent=4)
    
    print("Loading state dict into reconstructed model...")
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(keys.missing_keys)}")
    print(f"Unexpected keys: {len(keys.unexpected_keys)}")
    
    comp_flops, comp_params = calculate_theoretical_flops(model, "dbf")
    print(f"Compressed Model FLOPs/token: {comp_flops/1e9:.4f} GFLOPs")
    
    comp_ppl = eval_ppl(model, tokenizer)
    comp_speed, comp_mem = measure_speed_and_memory(model, tokenizer)
    comp_size = get_model_size(checkpoint_path)
    
    results['compressed'] = {
        'ppl': comp_ppl,
        'speed': comp_speed,
        'memory': comp_mem,
        'size_mb': comp_size,
        'flops': comp_flops
    }
    
    print(f"Compressed Model PPL: {comp_ppl:.4f}")
    
    print("\n=== Results ===")
    print(f"Model: {model_name}")
    print(f"Base PPL: {base_ppl:.4f} | Speed: {base_speed:.2f} t/s | Mem: {base_mem:.2f} MB | FLOPs: {base_flops/1e9:.2f}G")
    print(f"DBF PPL:  {comp_ppl:.4f} | Speed: {comp_speed:.2f} t/s | Mem: {comp_mem:.2f} MB | Size: {comp_size:.2f} MB | FLOPs: {comp_flops/1e9:.2f}G")
    
    # Save results
    with open(f"results_{args.model}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate Plots
    plot_sensitivity(f"sensitivity_{args.model}.json", f"sensitivity_{args.model}.png")
    plot_bit_allocation(f"bit_allocation_{args.model}.json", f"bit_allocation_{args.model}.png")

    # Print Table
    print("\n### A. Quantitative Metrics (Table)")
    print("| Metric | Dense | DBF (Uniform) | DBF (Smart) |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| Perplexity | {base_ppl:.4f} | {comp_ppl:.4f} | N/A |")
    print(f"| Model Size (MB) | N/A | {comp_size:.2f} | N/A |")
    print(f"| FLOPs (G) | {base_flops/1e9:.2f} | {comp_flops/1e9:.2f} | N/A |")
    print(f"| Speed (tok/s) | {base_speed:.2f} | {comp_speed:.2f} | N/A |")
    print(f"| Peak Mem (MB) | {base_mem:.2f} | {comp_mem:.2f} | N/A |")

if __name__ == "__main__":
    main()
