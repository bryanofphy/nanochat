# Learning Notes: Rotary Positional Embeddings (RoPE)

## 1. The Core Concept
RoPE is a method for encoding positional information in Transformers by rotating Query (Q) and Key (K) vectors in a 2D plane. It solves the limitations of traditional "Absolute Position Embeddings" used in earlier models like GPT-2.

**Paper Reference:** *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

## 2. Why RoPE? (Absolute vs. Relative)
*   **Absolute Position (GPT-2/nanoGPT):** The model learns a unique vector $p_i$ for every index $i$.
    *   *Drawback:* It cannot handle sequences longer than those seen during training (it hasn't "seen" Position 2049).
*   **Relative Position (RoPE):** Information is encoded as the **angle of rotation**.
    *   *Benefit:* **Length Extrapolation**. The relationship between words 5 steps apart is defined by the angle difference $5\theta$, which is invariant to absolute position.

## 3. The Mathematical "Magic"
RoPE treats the query ($q$) and key ($k$) vectors as complex numbers in the 2D plane.
For a token at position $m$, we rotate its vector by angle $m\theta$:

$$ f(q, m) = q \cdot e^{im\theta} $$

### The Dot Product Cancellation
Attention is calculated via the dot product (real part of geometric product). The interaction between position $m$ and $n$ becomes:
$$ \langle q_m, k_n \rangle = \text{Re}(q \cdot e^{im\theta} \cdot (k \cdot e^{in\theta})^* ) $$
$$ = \text{Re}(q \cdot k^* \cdot e^{i(m-n)\theta}) $$

The absolute positions $m$ and $n$ disappear, leaving only the relative distance $(m-n)$.

## 4. Implementation in `nanochat`
In `nanochat/gpt.py` (Line ~50), RoPE is implemented efficiently using real-valued matrix operations on pairs of dimensions:

```python
def apply_rotary_emb(x, cos, sin):
    # Splits x into pairs (x1, x2)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

### Critical Rule: Q/K Only
*   **Queries (Q) and Keys (K):** Rotated to determine **where** the model should look (attention scores).
*   **Values (V):** **NOT** rotated. Rotating Values would distort the semantic content of the word (changing "King" to something else just because it appeared at index 50).

---
# Deep Dive Reference: `nanochat/gpt.py`

This document captures the structural and mathematical innovations in the core model.

## 1. High-Level Architecture
*   **The Master Dial:** Everything (width, heads, LR) scales from `--depth`.
    *   `model_dim = depth * 64` (Line 28 in `gpt.py`)
*   **Sliding Window (SSSL):** Optimized attention pattern.
    *   **S (Short):** 1024 token window.
    *   **L (Long):** 2048 (full) window.
    *   *Logic:* Alternates 3 short context layers with 1 long context layer to save compute. Defined in `_compute_window_sizes` (Line 245).

## 2. The Input Stage
*   **Embedding Norm:** `rmsnorm` is applied *immediately* after `wte`.
    *   *Why:* Ensures the starting signal for the residual stream is unit variance and stable.
*   **`x0` Shortcut:** The normalized embedding (`x0`) is saved at Line 333.
*   **Layer-wise Injection:** `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` (Line 335).
    *   *Initialization:* `x0_lambdas` init to `0.1` (Line 215), slowly gating in the shortcut.
    *   *Benefit:* Prevents signal decay in deep networks (ResNet logic).

## 3. The Attention Mechanism (`CausalSelfAttention`)
*   **Native Layout:** Uses `(B, T, H, D)` throughout.
    *   *Why:* Optimized for Flash Attention 3; avoids expensive `.transpose()` operations.
*   **ResFormer Value Embeddings:**
    *   `v = v + gate * ve` (Line 90).
    *   **Gate:** Input-dependent scalar `[0, 2]` calculated from first 32 channels. `ve_gate` layer at Line 77.
    *   *Benefit:* Adds massive parameter capacity (memory) at near-zero FLOP cost.
*   **QK Norm:** Query and Key vectors are normalized *after* RoPE (Line 95).
    *   *Why:* Stabilizes the softmax "temperature" in deep layers where dot products can explode.

## 4. The MLP & Block
*   **ReLU² Activation:** `F.relu(x).square()` (Line 129).
    *   *Logic:* Smooth at zero (unlike ReLU), cheaper than GeLU/SwiGLU.
*   **Pre-Norm Residual Path:**
    *   The "Clean Residual Stream" (the main addition path) never gets normalized.
    *   Norms only happen on the *branches* entering `attn` and `mlp`.

## 5. The Optimizer Setup
*   **Hybrid Strategy:**
    *   **AdamW:** For Embeddings and Scalars (no 2D structure).
    *   **Muon:** For 2D Projection Matrices.
*   **Learning Rate Advantage:** Muon allows a **30x higher LR** for matrices, driving the massive speedup in convergence.

---
# Module 2: The Muon Optimizer (`nanochat/optim.py`)

**Goal:** Understand how `nanochat` achieves 30x higher learning rates for matrix parameters.

## 1. The Core Concept: Orthogonal Updates
Standard optimizers like SGD or AdamW update weights by subtracting a scaled gradient: $W \leftarrow W - \eta \cdot G$.
*   **Problem:** For deep networks, repeated matrix multiplications can cause activations to explode or vanish if the singular values of $W$ drift away from 1.
*   **Muon's Solution:** It forces the *update step* itself to be an **orthogonal matrix**.
    *   An orthogonal matrix $U$ satisfies $U^T U = I$. It acts as a pure rotation (preserving magnitude).
    *   By forcing updates to be orthogonal, we can take much larger steps without destabilizing the network.

## 2. Newton-Schulz Iteration (The Engine)
Computing the true orthogonalization (via SVD) is too slow ($O(N^3)$) for the training loop. Muon uses an iterative approximation called **Newton-Schulz**.

In `optim.py`, this is implemented as the **"Polar Express"** variant (a more stable refinement).
**Reference:** *Polar Express: Efficient & Stable Orthogonalization* (Amsel et al., 2025). [arXiv:2505.16932](https://arxiv.org/abs/2505.16932)

```python
# Iteratively refine X to be orthogonal
for a, b, c in polar_express_coeffs[:ns_steps]:
    A = X.mT @ X  # Compute correlation
    B = b * A + c * (A @ A)
    X = a * X + X @ B  # Update X
```
*   **Input:** The gradient (plus momentum) $G$.
*   **Process:** Run the loop 5 times (`ns_steps=5`).
*   **Output:** An orthogonalized update matrix that points in the same direction as $G$ but has "perfect" spectral properties.

## 3. Variance Reduction (NorMuon)
Muon tracks a "second momentum" (like Adam's $v$), but with a twist to save memory and fit the matrix structure.

*   **AdamW:** Stores a full $(M, N)$ matrix for $v$. Memory cost: $O(MN)$.
*   **Muon (Factored):** Stores a compressed vector of size $(M, 1)$ or $(1, N)$. Memory cost: $O(M)$ or $O(N)$.
    *   *Logic:* It normalizes the update based on the RMS of the columns (or rows).
    *   *Code:* `second_momentum_buffer` in `muon_step_fused`.

## 4. Hybrid Architecture (AdamW + Muon)
You cannot use Muon for everything.
*   **Embeddings & Scalars:** These are 1D vectors or lookup tables. Orthogonalization doesn't apply. -> **Use AdamW.**
*   **Linear Layers (Projections):** These are 2D matrices ($W_q, W_k, W_v, W_{fc}, W_{proj}$). -> **Use Muon.**

This hybrid setup is handled in `setup_optimizer` (in `gpt.py`) and executed in `MuonAdamW.step`.

## 5. Distributed Training (`DistMuonAdamW`)
To train large models on 8xH100s, `nanochat` implements a custom sharded optimizer (ZeRO-2 style) manually:
1.  **Reduce-Scatter:** Split the massive gradient tensors across GPUs.
2.  **Shard Update:** Each GPU runs Newton-Schulz on just its small chunk of the parameters.
3.  **All-Gather:** GPUs share the updated weights.

*Benefit:* This overlaps communication with the heavy computation of the Newton-Schulz loop, hiding latency.

---
# Module 3: The Physics of Training (`scripts/base_train.py`)

**Goal:** Understand how `nanochat` auto-calculates hyperparameters instead of guessing them.

## 1. Auto-Horizon (Chinchilla Scaling)
Instead of manually setting "epochs", `nanochat` calculates the optimal training duration based on model size.

*   **Formula:** `target_tokens = ratio * num_scaling_params`
*   **The Ratio:** Defaults to `10.5` (a bit more aggressive than Chinchilla's `20`).
*   **Paper Reference:** *Training Compute-Optimal Large Language Models* (Hoffmann et al., 2022). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
*   **Why?** This ensures that if you double the model size (`--depth`), the training time automatically extends to feed it enough data.

## 2. Auto-Batch Size (Power Laws)
We don't set batch size arbitrarily. We use the "Power Laws" finding (from *Power Lines* paper) that larger models/longer runs can utilize larger batches.

*   **Formula:** $B_{opt} \propto D^{0.383}$
*   **Code:**
    ```python
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    ```
*   **Paper Reference:** *Power Laws for Neural Language Models* (Kaplan et al., 2020) and subsequent batch-size studies.
*   **Intuition:** As training gets longer ($D$ increases), gradients become noisier (you are fine-tuning). A larger batch size averages out this noise, allowing you to take stable steps.

## 3. Auto-Learning Rate
If we increase the batch size, we take fewer steps. To cover the same distance, we must move faster.

*   **Formula:** $\text{LR} \propto \sqrt{\text{BatchSize} / \text{RefBatchSize}}$
*   **Code:** `batch_lr_scale = batch_ratio ** 0.5`
*   **Why Sqrt?** Standard heuristic for AdamW. Linear scaling is used for SGD, but Adam's variance normalization makes square root scaling more appropriate.

## 4. Auto-Weight Decay
Weight decay prevents overfitting by keeping weights small. Wider models need stronger constraint.

*   **Finding:** Optimal weight decay scales with $\frac{1}{\text{Width}^2}$ (or more specifically, related to the ratio of tokens to parameters).
*   **Code:**
    ```python
    weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    ```
*   **Result:** This keeps the "effective" regularization constant across different model scales.

---
# Module 4: System Optimization

**Goal:** Understand how to saturate the H100 GPU and make inference fast.

## 1. FP8 Training (`nanochat/fp8.py`)
This module implements a minimal, tensor-wise FP8 training recipe that runs ~2x faster on H100s.

*   **Tensor-wise Scaling:** Instead of calculating a scale factor for every row (which is expensive), we calculate one scale for the whole tensor:
    `scale = FP8_MAX / max(abs(tensor))`
*   **The Dtypes:**
    *   **Weights/Inputs:** `e4m3fn` (4 exponent, 3 mantissa) -> Higher precision. Used for weights to preserve small differences.
    *   **Gradients:** `e5m2` (5 exponent, 2 mantissa) -> Wider dynamic range. Gradients can explode or be tiny, so we need range over precision.
*   **Custom Autograd:** `_Float8Matmul` is a custom function that:
    1.  **Forward:** Quantizes inputs -> Calls `torch._scaled_mm` (cuBLAS) -> Returns BF16 output.
    2.  **Backward:** Quantizes incoming gradients -> Calls `torch._scaled_mm` -> Returns BF16 gradients.

## 2. Flash Attention 3 (`nanochat/flash_attention.py`)
*   **The Layout Change:** Standard PyTorch uses `(B, H, T, D)`. FA3 natively supports `(B, T, H, D)`.
    *   *Benefit:* We avoid the `.transpose(1, 2)` operation, which is a pure memory copy. In `gpt.py`, we keep everything in `(B, T, H, D)` from start to finish.
*   **The Fallback:** If you are NOT on an H100 (Hopper), the code detects it and falls back to `SDPA`:
    ```python
    if not _use_fa3():
        # Manually transpose to standard layout
        q = q.transpose(1, 2)
        return F.scaled_dot_product_attention(q, ...)
    ```

## 3. The Inference Engine (`nanochat/engine.py`)
*   **Static KV Cache:** We don't `append` to a list during inference (which re-allocates memory constantly).
    *   We **pre-allocate** a giant tensor `(Layers, Batch, MaxSeqLen, Heads, Dim)` on the GPU.
    *   We use a pointer (`cache_seqlens`) to know where to write the next token.
*   **In-Place Updates:** The `flash_attn_with_kvcache` kernel writes the new K/V vectors directly into this pre-allocated memory, avoiding overhead.

---
# Module 5: Evaluation & Metrics (`nanochat/core_eval.py`)

**Goal:** How to prove the model is smart (Reasoning) vs. just good at predicting words (Loss).

## 1. The CORE Metric Strategy
Standard "Loss" (Perplexity) is a bad metric for capabilities because it depends on the tokenizer and data distribution. The CORE metric evaluates the model on **22 standardized tasks** (MMLU, HellaSwag, etc.).

## 2. Likelihood-Based Scoring
How do you test a "Next Token Predictor" on a multiple-choice question?
*   **Method:** You don't ask it to output "A" or "B".
*   **The Process:**
    1.  Construct 4 candidate sentences:
        *   `Context + "Paris"`
        *   `Context + "London"`
        *   `Context + "Berlin"`
    2.  Measure the **Loss** (surprise) of the model for each continuation.
    3.  **Prediction:** Pick the option with the **lowest loss** (highest probability).
    4.  **Score:** If (Predicted == Gold Label), score 1, else 0.

## 3. Engineering Implementation
*   **Jinja Templates:** `render_prompts_mc` uses templates to format questions consistently.
*   **Common Prefix Detection:** The code calculates exactly where the "Question" ends and the "Answer" begins (`find_common_length`).
    *   *Why:* We only sum the loss over the **Answer** tokens. We don't penalize the model for finding the *Question* surprising.
*   **Distributed Eval:** The task is embarrassingly parallel. The script splits the 1000s of questions across the 8 GPUs (`rank`-based striding) and uses `all_reduce` to sum the final score.

---
# Module 6: Data Efficiency (`nanochat/dataloader.py`)

**Goal:** How to feed the beast (GPU) without choking it (padding) or confusing it (fragmentation).

## 1. The Problem
Training on raw text documents has a conflict:
*   **Documents have variable lengths** (Tweets are short, Wikipedia articles are long).
*   **GPUs need fixed shapes** (e.g., `(Batch=32, Seq=2048)`).

**Common Bad Solutions:**
*   **Padding:** Pad every document to 2048. *Result:* Massive compute waste (calculating attention on `0`s).
*   **Simple Concatenation:** Glue all docs together and chop arbitrarily at 2048. *Result:* A row might start in the middle of a sentence. The model has no context for the first 500 tokens.

## 2. The `nanochat` Solution: BOS-Aligned Best Fit
We use a **Bin Packing** algorithm (specifically **Best Fit Decreasing**) to fill the fixed-size rows.

### The Rules:
1.  **Hard Start:** Every row *must* start with a `<|bos|>` token.
    *   *Benefit:* The model always has a clean "reset" state at index 0. No "leaking" context from previous unrelated documents.
2.  **Best Fit Packing:**
    *   We maintain a buffer of ~1000 incoming documents.
    *   **Algorithm:**
        ```python
        while space_remaining > 0:
            # 1. Search buffer for largest doc that fits in space_remaining
            doc = find_largest_fitting_doc()
            if doc:
                row.append(doc)
            else:
                # 2. If nothing fits, crop a doc to fill the gap exactly
                doc = buffer.pop()
                row.append(doc[:space_remaining])
                # DISCARD the rest of the doc (Waste!)
        ```

### 3. The Trade-off
*   **Pros:**
    *   **100% Compute Utilization:** Zero padding tokens. Every FLOP learns something.
    *   **Clean Context:** Every token attends back to a valid start of document.
*   **Cons:**
    *   **Data Waste:** When we crop a document to fill the gap, the tail end is thrown away. In practice, this wastes **~35%** of the raw tokens.
    *   *Verdict:* Compute is the bottleneck, not data availability. Wasting cheap CommonCrawl tokens to save expensive H100 compute is a winning trade.

---
# Module 7: Research & Failures (`dev/LOG.md`)

**Goal:** Learn from things that *didn't* work.

## 1. Negative Result: Multi-Token Prediction (MTP)
*   **Idea:** Train the model to predict the next 2-4 tokens simultaneously using extra heads.
*   **Hypothesis:** It would force the model to "plan ahead" and learn deeper representations.
*   **Reality:** It failed to improve validation loss per wall-clock second.
    *   *Reason:* The extra parameters and compute for the MTP heads slowed down the main training loop more than the improved learning rate compensated for.

## 2. Negative Result: Varlen (Variable Length) Attention
*   **Idea:** Use complex masking to prevent attention from leaking across document boundaries when packing multiple docs in one row.
*   **Hypothesis:** "Cross-contamination" confuses the model.
*   **Reality:** Negligible improvement (< 0.0002 BPB).
    *   *Reason:* The model is smart enough to learn that `<|bos|>` is a reset. The complexity of Varlen kernels wasn't worth it.

## 3. Negative Result: SwiGLU
*   **Idea:** Use SwiGLU (standard in Llama) instead of MLP.
*   **Reality:** SwiGLU requires 3 matrix multiplications (Gate, Value, Output). `ReLU²` requires only 2 (Expansion, Output).
    *   For the specific "Time-to-GPT-2" speedrun, `ReLU²` was more efficient.

## 4. The Engineering Lesson
**"Papers are not Products."**
Many techniques that claim "State of the Art" results in papers (usually comparing "Steps to Convergence") often fail in the real world (comparing "Time to Convergence").
*   Paper metrics ignore: Python overhead, Kernel launch latency, Memory bandwidth, Distributed comms overhead.
*   `nanochat` optimizes for **Wall Clock Time**, which often means choosing simpler, faster algorithms (like `ReLU²` and Tensor-wise FP8) over theoretically "better" ones.
