# `nanochat` Core Library Documentation

This directory contains the implementation of the model, optimizer, and inference engine. It is designed to be **hackable** and **minimal**, avoiding the abstraction overhead of large frameworks.

## 1. `gpt.py`: The Model Architecture

This file implements the `GPT` class. While based on the standard Transformer decoder (GPT-2 style), it incorporates several modern improvements critical for performance and stability.

### Key Architectural Features & "Why"

*   **Rotary Positional Embeddings (RoPE)**
    *   **What:** Instead of adding absolute position vectors to the embeddings (like original GPT-2), RoPE rotates the Query and Key vectors by an angle proportional to their position in the sequence.
    *   **Why:** It allows the attention mechanism to understand *relative* distance between tokens naturally, which generalizes better to longer sequences than absolute embeddings.

*   **RMSNorm (Root Mean Square Normalization)**
    *   **What:** A simplified layer normalization that normalizes by the root mean square of activations, without centering the mean.
    *   **Why:** It is computationally cheaper than LayerNorm and provides equivalent convergence stability. In this repo, it is implemented *without* learnable parameters (affine=False) for further simplicity.

*   **SwiGLU vs. ReLU²**
    *   **Decision:** This model uses `Relu^2` (squared ReLU) instead of the popular SwiGLU.
    *   **Why:** Experiments (documented in `dev/LOG.md`) showed that for this specific scale/architecture, SwiGLU added parameter complexity without improving the "time-to-result" metric. Squared ReLU provides a similar non-linearity benefit (sharper activation) at lower cost.

*   **Learnable Scalars (`resid_lambdas`, `x0_lambdas`)**
    *   **What:**
        *   `x0_lambdas`: A skip connection directly from the initial embedding layer to every subsequent layer: $x_{layer} = \dots + \lambda_{x0} \cdot x_{embedding}$.
        *   `resid_lambdas`: A learnable gain on the residual branch.
    *   **Why:** Deep networks suffer from signal degradation. Direct access to the input embeddings (`x0`) helps gradients flow from the loss function all the way to the start of the network, improving training speed.

*   **Value Embeddings (ResFormer style)**
    *   **What:** Additional embedding tables that are added directly to the Value vectors in the attention mechanism.
    *   **Why:** This is a "cheat code" for increasing model capacity (parameter count) without increasing FLOPs (compute cost). It allows the model to store more "facts" or patterns without making the expensive matrix multiplications larger.

### Flash Attention 3 Integration
*   **What:** The model detects if it's running on an H100 (Hopper) GPU and uses `flash_attention.py` to call optimized FA3 kernels.
*   **Why:** Standard attention is $O(N^2)$ in memory and compute. Flash Attention is exact attention but tiled to fit in SRAM, reducing HBM (memory) access. FA3 optimizes this further for Hopper's asynchronous features, providing ~2x speedup over PyTorch's default SDPA.

---

## 2. `optim.py`: The Muon Optimizer

This is arguably the most novel part of the repo. Standard LLM training uses AdamW. This repo uses a hybrid **Muon + AdamW** approach.

### The Problem with AdamW
AdamW treats every parameter individually. For large 2D matrices (like the $W_q, W_k, W_v$ projections), this ignores the structural relationship between rows and columns.

### The Muon Solution
*   **Momentum Orthogonalized by Newton-schulz**: Muon updates 2D matrices by orthogonalizing them.
*   **Mechanism:** It essentially forces the weight update steps to be orthogonal transformations. This keeps the spectral radius of the weight matrices controlled, preventing exploding/vanishing gradients more effectively than simple gradient clipping.
*   **Newton-Schulz Iteration:** A method to compute the matrix inverse/orthogonalization iteratively on the GPU without costly SVD (Singular Value Decomposition).
*   **Result:** Matrix parameters converge significantly faster (in fewer steps) than with AdamW.

### The Hybrid Setup
*   **Embeddings & Scalars**: Trained with **AdamW**. (1D/Embedding tensors don't fit Muon's 2D logic).
*   **Matrix Weights**: Trained with **Muon**.
*   **Distributed**: `DistMuonAdamW` implements a custom distributed strategy (sharding optimizer state) similar to ZeRO-2, but optimized specifically to overlap communication (reduce-scatter/all-gather) with the heavy Muon compute.

---

## 3. `engine.py`: High-Performance Inference

Training is only half the story. `engine.py` implements the text generation loop.

### KV Cache (Key-Value Cache)
*   **Concept:** When generating the token at step $T$, you need the attention Keys and Values for tokens $0 \dots T-1$. Recomputing these every step is wasteful ($O(T^2)$).
*   **Implementation:** We calculate $K, V$ for the *new* token only, and append them to a pre-allocated buffer (`KVCache`).
*   **Memory Management:** The cache is pre-allocated on the GPU to avoid expensive memory allocations during the generation loop.

### Tool Use
*   The engine includes a simple state machine to detect special tokens (`<|python_start|>`).
*   It pauses generation, executes the Python code (e.g., a calculation), and inserts the result back into the context, allowing the model to "use tools" effectively.