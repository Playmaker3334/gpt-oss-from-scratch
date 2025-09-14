<!--
  GPT-OSS From Scratch — README
  Author: Rogelio Novelo (you can change this)
  License: MIT
-->

<h1 align="center">GPT-OSS From Scratch</h1>

<p align="center"><em>An educational, from-scratch implementation of a modern MoE Transformer inspired by GPT-OSS.</em></p>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-%23c4b5fd.svg?style=flat-square&labelColor=1f2937">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-%23c4b5fd.svg?style=flat-square&labelColor=1f2937">
  </a>
  <a href="./LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-%23c4b5fd.svg?style=flat-square&labelColor=1f2937">
  </a>
  <img alt="Model Size" src="https://img.shields.io/badge/Parameters-73.4M-%23c4b5fd.svg?style=flat-square&labelColor=1f2937">
  <img alt="Architecture" src="https://img.shields.io/badge/Architecture-MoE--Transformer-%23c4b5fd.svg?style=flat-square&labelColor=1f2937">
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#technical-details">Technical&nbsp;Details</a> •
  <a href="#training-results">Training&nbsp;Results</a> •
  <a href="#limitations">Limitations</a>
</p>

---

## Overview

This project implements a <strong>73.4M-parameter</strong> language model from scratch, incorporating modern innovations from recent transformer research. It is built to learn how today’s LLMs are structured and trained and to experiment with ideas like <em>Mixture of Experts (MoE), Grouped Query Attention (GQA), RoPE with YaRN,</em> and more.

> **Goal:** Provide a clear, hackable reference you can read, modify, and extend — not a production model.

> **Heads-up — Experimental.** This implementation is intended for learning and experimentation only. It trades production hardening for readability. Expect rough edges (e.g., minimal inference optimizations, small dataset, modest parameter count).

---

## Architecture

The model uses a **sparse Mixture-of-Experts (MoE) Transformer** with several modern optimizations.

### Core Components

| Component | Description | Configuration |
|---|---|---|
| **Mixture of Experts** | Sparse routing with top-k selection | 4 experts, 2 active per token |
| **Grouped Query Attention (GQA)** | Memory-efficient attention | 8 Q heads, 4 KV heads |
| **RoPE + YaRN** | Positional encoding with better length generalization | 2,048 context |
| **Attention Sinks** | Learned per-head biases for stability | per-head params |
| **SwiGLU** | Gated FFN for expressivity | fused impl |
| **RMSNorm** | Scale-invariant normalization | `eps=1e-5` |

### Model Configuration (Mini)

- **Total Parameters:** 73,369,088 (~73.4M)  
- **Transformer Layers:** 4  
- **Hidden Size:** 512  
- **Vocabulary:** 100,256 (tiktoken `cl100k_base`)  
- **Experts:** 4 (top-2 / token)  
- **Intermediate Size:** 768  
- **Max Seq Len:** 2048  
- **Attention:** 8 query heads, 4 key-value heads  

<details>
<summary><strong>Mermaid Overview (click)</strong></summary>

```mermaid
flowchart LR
  T[Token IDs] --> E[Token Embeddings]
  E --> B1[Transformer Block × 4]
  subgraph Block
    A[Attention (GQA + RoPE + Sinks)] --> N1[RMSNorm]
    N1 --> M[MoE (Top-2 of 4 Experts)]
    M --> N2[RMSNorm]
  end
  B1 --> H[Head / LM Logits] --> O[Softmax]
```
</details>

---

## Features

- **MoE** with token-choice routing, auxiliary load-balancing losses (importance + z-loss), and jitter noise.  
- **GQA** to reduce KV memory while maintaining performance.  
- **RoPE + YaRN** for length extrapolation.  
- **Attention sinks** for stability.  
- Clean **PyTorch 2.x** code, easy to read and extend.  
- Educational defaults suitable for a **single-GPU** or **dual-GPU (DDP)** experiment.

---

## Installation

### Requirements
- Python **3.8+**
- PyTorch **2.0+**
- CUDA **11.7+** (for GPU)
- **16GB VRAM** per GPU minimum recommended  
- Reference hardware for this repo: **2 × NVIDIA P100 (16GB)**

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt-oss-from-scratch.git
cd gpt-oss-from-scratch/gpt_oss_20b

# Install deps
pip install "torch>=2.0" tiktoken numpy tqdm
```

---

## Usage

### 1) Data Preparation

```bash
# Prepare a small OpenWebText sample (~150MB)
python prep_corpus.py
```

### 2) Training (single GPU)

```bash
python main.py \
  --train_data train.txt \
  --eval_data eval.txt \
  --num_epochs 3 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --learning_rate 3e-4 \
  --warmup_steps 500 \
  --gradient_checkpointing
```

### 2b) Training (dual GPU via DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py \
  --train_data train.txt \
  --eval_data eval.txt \
  --num_epochs 3 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --learning_rate 3e-4 \
  --warmup_steps 500 \
  --gradient_checkpointing
```

> Notes: P100 does not provide Tensor Cores; mixed precision (AMP) may not speed up training. Leave FP32 as default or benchmark AMP cautiously.

### 3) Text Generation (toy)

```python
from src.model.gpt_oss import GPTOSSForCausalLM
from src.tokenizer.harmony_tokenizer import HarmonyTokenizer
from config.model_config import GPTOSSConfigMini

# Load configuration
config = GPTOSSConfigMini()

# Initialize model
model = GPTOSSForCausalLM(
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    num_attention_heads=config.num_attention_heads,
    num_key_value_heads=config.num_key_value_heads,
    intermediate_size=config.intermediate_size,
    num_experts=config.num_experts,
    num_experts_per_token=config.num_experts_per_token,
)

# Load tokenizer
tokenizer = HarmonyTokenizer()

# Generate text
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
generated = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)
print(tokenizer.decode(generated[0]))
```

---

## Project Structure

```
gpt_oss_20b/
├── config/
│   └── model_config.py              # Model configuration
├── src/
│   ├── model/
│   │   ├── gpt_oss.py               # Main model implementation
│   │   ├── transformer_block.py     # Transformer layers
│   │   ├── attention/
│   │   │   ├── grouped_query_attention.py
│   │   │   ├── rope.py              # Rotary position embeddings
│   │   │   └── attention_sinks.py   # Attention sink implementation
│   │   ├── moe/
│   │   │   ├── mixture_of_experts.py
│   │   │   ├── router.py            # Expert routing logic
│   │   │   └── expert.py            # Individual expert networks
│   │   ├── feedforward/
│   │   │   └── swiglu.py            # SwiGLU activation
│   │   ├── embeddings/
│   │   │   └── token_embeddings.py  # Token embeddings
│   │   └── normalization/
│   │       └── rmsnorm.py           # RMS normalization
│   ├── tokenizer/
│   │   └── harmony_tokenizer.py     # Tokenizer wrapper
│   └── utils/
│       ├── tensor_utils.py          # Tensor operations
│       └── math_utils.py            # Mathematical utilities
├── main.py                          # Training script
└── prep_corpus.py                   # Data preparation
```

---

## Technical Details

### Mixture of Experts

- Token-choice routing with softmax normalization  
- Auxiliary losses for load balancing (importance + z-loss)  
- Expert-parallelism-friendly design  
- Router jitter noise during training for exploration  
- Efficient batching of expert computations

### Attention (GQA)

- 2:1 ratio of query to key-value heads  
- Optional sparse / sliding-window patterns  
- Attention sinks to stabilize training  
- Flash-Attention–ready architecture

### Position Embeddings

- RoPE with YaRN extension for better length generalization  
- Base theta: 1,000,000 for extended context  
- NTK-aware interpolation  
- Dynamic positional indices

---

## Training Results

| Metric | Value | Context |
|---|---|---|
| Final Loss | 7.5 | expected for this scale / data |
| Perplexity | ~1900 | typical for ~73M params on ~150MB |
| Hardware | 2 × NVIDIA P100 (16GB) | FP32 by default; AMP optional |


For reference, **GPT-2 (124M)** trained on **~40GB** achieves perplexity ~30 — orders of magnitude more data and scale.

---

## Comparison with GPT-OSS

| Feature | GPT-OSS (20B) | This Repo | Scale |
|---|---:|---:|---:|
| Parameters | 20B | 73.4M | 1/272× |
| Experts | 128 | 4 | 1/32× |
| Hidden Size | 2880 | 512 | 1/5.6× |
| Layers | 36 | 4 | 1/9× |
| Vocabulary | 201,088 | 100,256 | 1/2× |
| GQA Ratio | 8:1 | 2:1 | — |
| Architecture | MoE Transformer | MoE Transformer | Same |

---

## Limitations

- Scale: 73.4M parameters — do not expect complex reasoning or long-form coherence.  
- Data: trained on ~150MB (vs. TBs for production models).  
- Optimization: lacks production features (quantization, custom kernels, etc.).  
- Purpose: learning and experimentation — not deployment.

---

## Roadmap

- Add Flash-Attention and fused kernels  
- BF16 / FP8 training support  
- 4-bit / 8-bit quantized inference  
- Longer context (4k–8k)  
- More robust evaluation harness

---

## Citation

```bibtex
@misc{openai2025gptoss,
  title        = {gpt-oss-120b \& gpt-oss-20b Model Card},
  author       = {OpenAI},
  year         = {2025},
  eprint       = {2508.10925},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.10925}
}
```

---

## License

MIT License — see [LICENSE](./LICENSE).

---

