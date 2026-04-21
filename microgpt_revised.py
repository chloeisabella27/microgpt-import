# =============================================
# microGPT (Modified) with:
# - GELU
# - LoRA
# - RoPE
# - Mixture of Experts (MoE)
# =============================================

import math
import random
random.seed(42)

# -----------------------------
# Autograd Value class (unchanged)
# -----------------------------
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0

    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)

    def __repr__(self):
        return f"Value({self.data})"

# -----------------------------
# Basic utilities
# -----------------------------
def softmax(logits):
    max_val = max(x.data for x in logits)
    exps = [Value(math.exp(x.data - max_val)) for x in logits]
    total = sum(e.data for e in exps)
    return [Value(e.data / total) for e in exps]

# -----------------------------
# GELU Activation
# -----------------------------
def gelu(x):
    return Value(x.data * 0.5 * (1 + math.tanh(math.sqrt(2/math.pi) * (x.data + 0.044715 * x.data**3))))

# -----------------------------
# Linear layer
# -----------------------------
def linear(x, w):
    return [Value(sum(wi.data * xi.data for wi, xi in zip(row, x))) for row in w]

# -----------------------------
# LoRA Linear Layer
# -----------------------------
def linear_lora(x, w, A, B, alpha=1.0):
    base = linear(x, w)

    # Low-rank update
    Ax = [sum(ai.data * xi.data for ai, xi in zip(a_row, x)) for a_row in A]
    BAx = [sum(bi.data * Ax_i for bi, Ax_i in zip(b_row, Ax)) for b_row in B]

    return [Value(b.data + alpha * l) for b, l in zip(base, BAx)]

# -----------------------------
# RoPE (Rotary Position Embedding)
# -----------------------------
def apply_rope(x, pos):
    out = []
    d = len(x)
    for i in range(0, d, 2):
        theta = 1.0 / (10000 ** (i / d))
        cos_t = math.cos(pos * theta)
        sin_t = math.sin(pos * theta)

        x1 = x[i].data
        x2 = x[i+1].data if i+1 < d else 0

        out.append(Value(x1 * cos_t - x2 * sin_t))
        out.append(Value(x1 * sin_t + x2 * cos_t))

    return out

# -----------------------------
# Mixture of Experts (MoE)
# -----------------------------
print("Output:", out)
