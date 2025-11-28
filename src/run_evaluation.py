import os
import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluate import (
    eval_ppl, 
    measure_speed_and_memory, 
    calculate_theoretical_flops, 
    plot_sensitivity, 
    plot_bit_allocation,
    reconstruct_model,
    get_model_size
)
from src.modelutils import find_layers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to compressed checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--sensitivity_file', type=str, default=None, help='Path to sensitivity json')
    parser.add_argument('--bit_allocation_file', type=str, default=None, help='Path to bit allocation json')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu" # Load on CPU first
    )
    
    # 1. Dense Metrics (Baseline)
    # We can run these if we want, but maybe user just wants compressed metrics.
    # Let's assume we want compressed metrics primarily.
    # But for FLOPs reduction, we need Dense FLOPs.
    
    print("Calculating Dense FLOPs...")
    dense_flops, _ = calculate_theoretical_flops(model, model_type="dense")
    print(f"Dense FLOPs: {dense_flops:.4e}")
    
    # 2. Load Compressed Checkpoint
    print(f"Loading compressed checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    
    # 3. Reconstruct Model
    model, bit_alloc = reconstruct_model(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to("cuda")
    
    # 4. Compressed Metrics
    print("Calculating DBF FLOPs...")
    dbf_flops, _ = calculate_theoretical_flops(model, model_type="dbf")
    print(f"DBF FLOPs: {dbf_flops:.4e}")
    
    flops_reduction = dense_flops / dbf_flops
    print(f"FLOPs Reduction: {flops_reduction:.2f}x")
    
    # 5. PPL
    ppl = eval_ppl(model, tokenizer, device="cuda")
    
    # 6. Speed & Memory
    speed, memory = measure_speed_and_memory(model, tokenizer, device="cuda")
    
    # 7. Model Size
    size_mb = get_model_size(args.checkpoint)
    
    # 8. Plots
    if args.sensitivity_file:
        plot_sensitivity(args.sensitivity_file, os.path.join(args.output_dir, "sensitivity.png"))
    
    if args.bit_allocation_file:
        plot_bit_allocation(args.bit_allocation_file, os.path.join(args.output_dir, "bit_allocation.png"))
        
    # Save Report
    report = {
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "ppl": ppl,
        "speed_tokens_sec": speed,
        "peak_memory_mb": memory,
        "size_mb": size_mb,
        "dense_flops": dense_flops,
        "dbf_flops": dbf_flops,
        "flops_reduction": flops_reduction
    }
    
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # Markdown Report
    md_report = f"""
# Evaluation Report: {args.model_name}

## Metrics
| Metric | Value |
| :--- | :--- |
| **Perplexity (PPL)** | {ppl:.2f} |
| **Inference Speed** | {speed:.2f} tokens/sec |
| **Peak Memory** | {memory:.2f} MB |
| **Model Size** | {size_mb:.2f} MB |
| **Theoretical FLOPs (Dense)** | {dense_flops:.2e} |
| **Theoretical FLOPs (DBF)** | {dbf_flops:.2e} |
| **FLOPs Reduction** | {flops_reduction:.2f}x |

## Visualizations
![Sensitivity](sensitivity.png)
![Bit Allocation](bit_allocation.png)
"""
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write(md_report)
        
    print("Evaluation complete. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()
