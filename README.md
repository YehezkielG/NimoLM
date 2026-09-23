# NimoLM

NimoLM is a collection of language models developed from scratch to explore efficient language-model architectures, training techniques, and model scaling.

## 🚀 NimoLM-244M

NimoLM-244M is a 244M-parameter decoder-only language model currently under development.

The model is designed to explore a hybrid architecture combining **Gated Linear Attention (GLA)** with conventional **causal self-attention**, aiming to investigate the trade-offs between computational efficiency and language-modeling performance.

### Architecture

The current NimoLM-244M architecture includes:

- Decoder-only language model
- Hybrid Gated Linear Attention (GLA) + causal self-attention
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU
- Custom tokenizer
- Autoregressive next-token prediction
- PyTorch-based training pipeline

### Model Overview

| Component | NimoLM-244M |
|---|---|
| Architecture | Decoder-only |
| Parameters | ~244M |
| Attention | Hybrid GLA + causal self-attention |
| Positional Encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Framework | PyTorch |
| Training | From scratch |

## 🧠 Why NimoLM?

Standard self-attention provides strong token-to-token interaction but has quadratic complexity with respect to sequence length.

NimoLM explores **Gated Linear Attention** as an alternative mechanism that can process information more efficiently while retaining useful contextual representations.

The hybrid architecture combines both approaches rather than replacing self-attention entirely.

This project is primarily an engineering and research exploration into:

- Efficient attention mechanisms
- Hybrid attention architectures
- Language-model pretraining
- Transformer architecture design
- Training stability
- Model efficiency at moderate parameter scales

## 📂 Repository Structure

```text
NimoLM/
├── nimo_src/
│   └── ...
├── model-collection/
│   └── nimo244m/
├── gla.py
├── pyproject.toml
└── README.md
