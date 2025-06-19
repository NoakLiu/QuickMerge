# QuickMerge++

A general-purpose token compression framework designed to accelerate autoregressive (AR) generative models across text, image, and video modalities.

## Overview

QuickMerge++ introduces three key innovations for efficient token compression:

1. **Entropy-aware saliency estimation** via attention distributions across layers
2. **Differentiable token merging** with norm-guided foreground selection  
3. **Bidirectional autoregressive alignment** to preserve decoding consistency post-compression

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from quickmerge import QuickMergePP

# Initialize QuickMerge++
embedding_dim = 512
quickmerge = QuickMergePP(embedding_dim=embedding_dim)

# Input embeddings [batch_size, num_tokens, embedding_dim]
X = torch.randn(2, 100, 512)
target_tokens = 20

# Compress tokens
merged_tokens, losses = quickmerge(X, target_tokens)
print(f"Compressed from {X.shape[1]} to {merged_tokens.shape[1]} tokens")
```

## Core Components

### 1. Entropy-Aware Saliency Estimation

Computes token importance using normalized attention entropy across Transformer layers:

```python
from quickmerge import EntropyAwareSaliency

saliency_estimator = EntropyAwareSaliency(embedding_dim=512, num_layers=12)
saliency_scores = saliency_estimator(X)  # [B, N]
```

### 2. Differentiable Token Merging

Uses Gumbel-Softmax for end-to-end optimization of token selection:

```python
from quickmerge import DifferentiableTokenMerging

token_merger = DifferentiableTokenMerging(temperature=0.1, epsilon=0.01)
merged_tokens, mask = token_merger(X, saliency_scores, K=20)
```

### 3. Bidirectional AR Alignment

Ensures compressed sequences remain valid for autoregressive decoding:

```python
from quickmerge import BidirectionalARAlignment

ar_alignment = BidirectionalARAlignment(embedding_dim=512)
alignment_loss = ar_alignment.compute_alignment_loss(merged_tokens)
```

### 4. Norm-Based Fidelity Constraint

Preserves semantic content through norm-mass retention:

```python
from quickmerge import NormBasedFidelityConstraint

fidelity_constraint = NormBasedFidelityConstraint(gamma=0.8)
fidelity_loss = fidelity_constraint.compute_fidelity_loss(X, merged_tokens)
```

## Inference Pipeline

```python
from quickmerge import quickmerge_inference

# Single sequence inference
X_single = torch.randn(100, 512)  # [N, D]
ar_model = YourARModel()  # Your autoregressive model

merged_tokens, predictions = quickmerge_inference(
    X_single, 
    ar_model, 
    entropy_budget=0.2  # Compress to 20% of original tokens
)
```

## Algorithm

The QuickMerge++ inference pipeline follows these steps:

1. **Compute saliency** via attention entropy across layers
2. **Sample mask** via Gumbel-softmax with temperature τ
3. **Assign merge weights** using saliency mass
4. **Cluster tokens** into K groups using cosine similarity
5. **Compute merged tokens** via saliency-weighted averaging
6. **Perform left-to-right decoding** on compressed sequence

## Mathematical Formulation

### Saliency Score
```
s_i = (1/L) * Σ_l Normalize(H_i^(l))
```
where `H_i^(l)` is the attention entropy at layer l.

### Token Merging
```
π_i = exp((s_i + g_i)/τ) / Σ_j exp((s_j + g_j)/τ)
x̃_k = Σ_{j∈G_k} (m̃_j * x_j) / Σ_{j'∈G_k} m̃_{j'}
```

### AR Alignment Loss
```
L_AR = L_forward + L_backward
```

## Example Usage

See `example.py` for a complete demonstration:

```bash
python example.py
```

## Parameters

- `embedding_dim`: Dimension of input embeddings
- `num_layers`: Number of Transformer layers for saliency computation
- `temperature`: Gumbel-Softmax temperature (lower = more discrete)
- `gamma`: Norm-mass retention ratio for fidelity constraint
- `epsilon`: Small constant for background token weights
- `alpha`: Weight for AR alignment loss

## Citation

If you use QuickMerge++ in your research, please cite:

```bibtex
@article{quickmergepp2024,
  title={QuickMerge++: A General-Purpose Token Compression Framework for Autoregressive Models},
  author={Dong Liu and Yanxuan Yu},
  journal={ICML 2025},
  year={2025}
}
```

## License

MIT License 