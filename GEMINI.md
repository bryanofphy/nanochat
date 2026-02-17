# Gemini Project Context

- **User Profile**: Engineer focused on learning efficient LLM training techniques.
- **Project Structure**: The `nanoGPT` repository has been copied into this folder at `./nanoGPT/` for direct comparison with `nanochat`.
- **Learning Goals**: Understanding the transition from AdamW-based training (nanoGPT) to modern, highly optimized training using Muon, Flash Attention 3, and specialized scaling laws (nanochat).

## 🎓 Technical Deep Dive Study Plan

This curriculum moves from the "Mathematical Core" outward to "System Architecture."

### 🏛️ Module 1: Modern Architecture (`nanochat/gpt.py`)
**Goal:** Understand the "Free Lunches" — architectural changes that improve performance without adding FLOPs.
*   **RoPE (Rotary Positional Embeddings):**
    *   *Concept:* Replacing absolute learned embeddings with relative rotations.
    *   *Code Focus:* `apply_rotary_emb` function and its usage inside `CausalSelfAttention.forward`.
    *   *Question:* How does `x1 * cos + x2 * sin` implement a rotation? Why is `n_head` relevant here?
*   **RMSNorm vs. LayerNorm:**
    *   *Concept:* Removing the mean-centering operation to save compute.
    *   *Code Focus:* The `norm(x)` function using `F.rms_norm`.
    *   *Question:* Where are the learnable parameters ($\gamma, \beta$)? (Hint: `nanochat` removed them).
*   **Value Embeddings:**
    *   *Concept:* Adding "memory" capacity without increasing matrix multiplication cost.
    *   *Code Focus:* `self.value_embeds` in `GPT` and the gating logic in `CausalSelfAttention`.
    *   *Question:* How are `ve` (value embeddings) added to `v` (values) in the attention head?
*   **ReLU² vs. SwiGLU:**
    *   *Code Focus:* `MLP` class.
    *   *Context:* See `dev/LOG.md` (Feb 5, 2026) for why SwiGLU was rejected.
*   **The `x0` Shortcut:**
    *   *Code Focus:* `GPT.forward` loop.
    *   *Question:* How does `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` help gradient flow?

### ⚡ Module 2: The Muon Optimizer (`nanochat/optim.py`)
**Goal:** Master the algorithm allowing 30x higher learning rates.
*   **Newton-Schulz Iteration:**
    *   *Code Focus:* `muon_step_fused` function.
    *   *Logic:* Trace the `for` loop using `polar_express_coeffs`. This is the "orthogonalization" step.
*   **Variance Reduction (NorMuon):**
    *   *Code Focus:* Look for `second_momentum_buffer` and the `v_norm` calculations inside `muon_step_fused`.
    *   *Question:* Why do we need to normalize columns differently?
*   **Parameter Grouping:**
    *   *Code Focus:* `GPT.setup_optimizer`.
    *   *Question:* Why are `matrix_params` separated from `embedding_params`?

### 🧪 Module 3: The Physics of Training (`scripts/base_train.py`)
**Goal:** Stop guessing hyperparameters. Learn to *calculate* them.
*   **Scaling Laws (Chinchilla):**
    *   *Code Focus:* Calculation of `target_tokens` using `target_param_data_ratio` (~10.5).
*   **Power Laws (Batch Size):**
    *   *Code Focus:* The formula `total_batch_size = 2 ** round(math.log2(B_REF * ...))`
    *   *Math:* $B_{opt} \propto D^{0.383}$. Larger models need larger batches.
*   **Weight Decay Scaling:**
    *   *Code Focus:* `weight_decay_scaled` calculation.
    *   *Math:* $WD \propto 1/width^2$.
*   **Learning Rate Scaling:**
    *   *Code Focus:* `batch_lr_scale` calculation. Why $\sqrt{BatchSize}$?

### 💾 Module 4: System Optimization (`nanochat/flash_attention.py` & `nanochat/engine.py`)
**Goal:** Saturate the H100 GPU.
*   **Flash Attention 3:**
    *   *Code Focus:* `flash_attn.flash_attn_func`.
    *   *Concept:* Understand the `(B, T, H, D)` memory layout. Why does avoiding `.transpose()` matter?
*   **KV Cache:**
    *   *Code Focus:* `KVCache` class in `engine.py`.
    *   *Question:* How does `prefill` work to speed up the first step of generation?
*   **FP8 Training:**
    *   *Code Focus:* `scripts/base_train.py`.
    *   *Context:* Read the "Microbenchmark vs Reality" section in `dev/LOG.md` (Jan 13, 2026).

### 📦 Module 5: Data Efficiency (`nanochat/dataloader.py`)
**Goal:** Feed the beast without choking it.
*   **BOS-Aligned Bin Packing:**
    *   *Code Focus:* `tokenizing_distributed_data_loader_with_state_bos_bestfit`.
    *   *Algorithm:* Trace the `while` loop that searches `doc_buffer` for the "Best Fit" document to fill the row.
    *   *Trade-off:* 100% utilization vs. cropping waste.

### 📓 Module 6: The Research Log (`dev/LOG.md`)
**Goal:** Learn from failure (Engineering Wisdom).
*   **Negative Results:**
    *   Why did **Multi-Token Prediction (MTP)** fail? (Jan 12 entry)
    *   Why did **Varlen Attention** fail? (Jan 13 entry)
*   **Lesson:** Understanding that "SOTA" techniques from papers often don't translate to wall-clock speedups in practice.
