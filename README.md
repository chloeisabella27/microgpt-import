# MicroGPT Extensions: GELU, LoRA, RoPE, MoE

## 1. GELU (Gaussian Error Linear Unit)
GELU is an activation function defined as:

    GELU(x) = x Φ(x)

where Φ(x) is the Gaussian cumulative distribution function.

Unlike ReLU, GELU smoothly weights inputs instead of hard-thresholding them.
This improves gradient flow and model performance.

---

## 2. LoRA (Low-Rank Adaptation)
LoRA modifies weight updates as:

    W = W0 + BA

where A and B are low-rank matrices.

Instead of updating the full matrix W, only A and B are trained.

Benefits:
- Massive reduction in trainable parameters
- No inference slowdown
- Efficient fine-tuning

---

## 3. RoPE (Rotary Position Embedding)
RoPE encodes positional information by rotating embeddings in vector space.

Key idea:
- Each pair of dimensions is rotated based on token position
- Enables attention to depend on relative positions

Benefits:
- Better long-range dependency modeling
- Works naturally with attention

---

## 4. Mixture of Experts (MoE)
MoE replaces a single MLP with multiple expert networks:

    y = sum(g_i(x) * Expert_i(x))

where g_i(x) is a gating function.

Benefits:
- Increased model capacity
- Sparse computation
- Better scaling

---

## Summary
This project integrates four modern Transformer improvements into a minimal GPT implementation:

- GELU improves activation smoothness
- LoRA enables efficient adaptation
- RoPE improves positional encoding
- MoE increases model capacity

Together, these modifications reflect techniques used in modern large language models.
