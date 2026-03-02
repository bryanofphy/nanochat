# Gemini Project Context: Mastering `nanochat`

- **User Profile**: Engineer focused on learning efficient LLM training techniques.
- **Instruction**: ALWAYS use the default values defined in this repository (e.g., `n_embd=768`, `n_head=6`) when explaining concepts. If a generic example is used, explicitly state so to avoid confusion.
- **Goal**: Understand the transition from "Vanilla Transformer" to "State-of-the-Art Efficient LLM" ($72 GPT-2).

## 🎓 Technical Deep Dive Study Plan

This curriculum moves from the "Mathematical Core" outward to "System Architecture" and "Evaluation."

### 🏛️ Module 1: Modern Architecture (`nanochat/gpt.py`)
**Goal:** Understand the "Free Lunches" and structural choices.
*   **RoPE (Rotary Positional Embeddings):**
    *   *Concept:* Relative vs. Absolute position.
    *   *Code:* `apply_rotary_emb` and `CausalSelfAttention`.
*   **RMSNorm & Pre-Norm:**
    *   *Concept:* Normalizing *before* the block. Why `nanochat` has an extra norm at the start and no final norm?
    *   *Code:* `rmsnorm` usage in the `forward` loop.
*   **Channels & Width:**
    *   *Concept:* `n_embd = depth * 64`. How the model dimensions are derived.
*   **Value Embeddings:**
    *   *Concept:* Adding "memory" capacity without compute cost.
    *   *Code:* `self.value_embeds` and the gating logic.
*   **ReLU²:**
    *   *Concept:* Why Squared ReLU beat SwiGLU for this specific scale.

### ⚡ Module 2: The Muon Optimizer (`nanochat/optim.py`)
**Goal:** Master the algorithm allowing 30x higher learning rates.
*   **Newton-Schulz Iteration:**
    *   *Code:* `muon_step_fused`.
    *   *Math:* How iterative orthogonalization works on the GPU.
*   **Variance Reduction (NorMuon):**
    *   *Code:* `second_momentum_buffer`.
    *   *Why:* Normalizing columns vs. rows.

### 🧪 Module 3: The Physics of Training (`scripts/base_train.py`)
**Goal:** Stop guessing hyperparameters. Learn to *calculate* them.
*   **Scaling Laws (Chinchilla):**
    *   *Code:* `target_tokens = ratio * num_scaling_params`.
    *   *Concept:* The 20:1 ratio vs. Inference-Optimality.
*   **Power Laws (Batch Size):**
    *   *Code:* `total_batch_size` calculation based on $D^{0.383}$.
*   **Weight Decay Scaling:**
    *   *Code:* `weight_decay_scaled`. Why $WD \propto 1/width^2$.

### 💾 Module 4: System Optimization
**Goal:** Saturate the H100 GPU.
*   **FP8 Training (`nanochat/fp8.py`):**
    *   *Concept:* Tensor-wise scaling.
    *   *Code:* `_Float8Matmul` and `_to_fp8`. How we use `torch._scaled_mm`.
*   **Flash Attention 3 (`nanochat/flash_attention.py`):**
    *   *Code:* `flash_attn_func`.
    *   *Concept:* Memory layout `(B, T, H, D)` and avoiding transposes.
*   **KV Cache (`nanochat/engine.py`):**
    *   *Code:* `KVCache`. The `.append()` logic for inference.

### 📊 Module 5: Evaluation & Metrics (`nanochat/core_eval.py`)
**Goal:** Prove intelligence, don't just memorize.
*   **The CORE Metric:**
    *   *Code:* `evaluate_example` and `render_prompts`.
    *   *Concept:* How we turn "next token prediction" into a "multiple choice exam."
*   **BPB (Bits Per Byte):**
    *   *Concept:* Standardizing loss across different tokenizers.

### 📦 Module 6: Data Efficiency (`nanochat/dataloader.py`)
**Goal:** Feed the beast without choking it.
*   **BOS-Aligned Bin Packing:**
    *   *Code:* `tokenizing_distributed_data_loader_with_state_bos_bestfit`.
    *   *Concept:* 100% GPU utilization vs. context fragmentation.

### 📓 Module 7: Research & Failures (`dev/LOG.md`)
**Goal:** Engineering Wisdom.
*   **Negative Results:** Why MTP (Multi-Token Prediction) and Varlen Attention failed to pay for themselves.
*   **The Speedrun:** Understanding the $72 milestone.
