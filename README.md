# Double Binary Factorization for Lightweight LLMs

Practical implementation of DBF compression on OPT-125M and TinyLlama-1.1B with Smart non-uniform extension.

## Quick Start

```bash
# 1. Setup environment
conda create --name dbf_env python=3.10 -y
conda activate dbf_env
pip install 'torch>=2.6' torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Run compression (choose one)
python -m src.compress_opt           # OPT-125M uniform (2-bit, ~2 hours)
python -m src.compress_smart_opt     # OPT-125M smart (variable bits, ~2.5 hours)
python -m src.compress_tinyllama     # TinyLlama-1.1B uniform (2-bit, ~8 hours)

# 3. Evaluate results (pass appropriate arguments based on above code output)
python -m src.run_evaluation         # Perplexity, speed, memory metrics
```

## Code Structure

```
DA422_CourseProject/
├── src/
│   ├── dbf_core.py              # Core DBF: ADMM optimization, SVID projection, factorization
│   ├── layers.py                # BitLinear layer, binary matrix operations
│   ├── modelutils.py            # Model loading, layer extraction utilities
│   ├── datautils.py             # Wikitext-2 calibration data loading
│   ├── compress_opt.py          # OPT-125M uniform compression (2-bit)
│   ├── compress_smart_opt.py    # OPT-125M smart compression (variable k)
│   ├── compress_tinyllama.py    # TinyLlama-1.1B uniform compression
│   ├── get_sensitivity.py       # Compute layer-wise gradient sensitivity
│   ├── evaluate.py              # Perplexity evaluation on Wikitext-2
│   └── run_evaluation.py        # Full evaluation suite (PPL, speed, memory)
├── README.md                    # This file
└── ProjectProposal.tex          # Original Project Proposal
```

## Reproduction Results

**OPT-125M (Uniform DBF):**
- Compression: 230.8 MB → 92.9 MB (2.48×)
- Perplexity: 23.06 → 42.10 PPL (+19.0)
- Speed: 22.0 → 81.4 tok/s (3.7× faster)

**OPT-125M (Smart DBF):**
- Compression: 230.8 MB → 93.8 MB (2.46×)
- Perplexity: 23.06 → 41.60 PPL (+18.5, 0.5 PPL improvement vs Uniform)
- Variable k: 380-1224 across 72 layers based on sensitivity

**TinyLlama-1.1B (Uniform DBF):**
- Compression: 2019.6 MB → 427.6 MB (4.72×)
- Perplexity: 6.96 → 11.06 PPL (+4.1)
- Speed: 10.8 → 40.3 tok/s (3.73× faster)
- Memory: 2109 MB → 478 MB VRAM (fits on 4GB GPUs)

## Key Features

- **Flexible compression:** Adjust `--target-bits` (1.0-3.0) for any compression ratio
- **Sensitivity-based allocation:** Smart compression uses gradient norms to allocate variable bits per layer
- **Hardware efficiency:** 3.7× speedup via memory bandwidth optimization
- **Reproducible:** All results in `output/` with JSON metrics and plots

## Citation

Original paper: [Addition is almost all you need: Compressing neural networks with double binary factorization](https://arxiv.org/abs/2505.11076)