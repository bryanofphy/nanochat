# `scripts` Directory Documentation

This directory contains the operational logic for training and evaluating the model. The flagship script is `base_train.py`, which is an executable lecture on modern LLM training heuristics.

## `base_train.py`: Auto-Configuring Training Loop

Unlike typical training scripts where you manually set batch size, learning rate, and iterations, this script **derives** them from first principles and scaling laws.

### 1. Scaling Laws (Chinchilla & Kaplan)
**Concept:** There is an optimal ratio of "Training Data" to "Model Size".
*   **Chinchilla (Hoffmann et al.)**: Suggests $\text{Tokens} \approx 20 \times \text{Params}$.
*   **Nanochat logic:** Uses a slightly more aggressive ratio (default ~10.5) derived from specific experiments on this architecture (see `dev/LOG.md`).
*   **Implementation:**
    ```python
    target_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = target_tokens // total_batch_size
    ```
    This guarantees that if you double the model size, the training duration scales appropriately to make it compute-optimal.

### 2. Batch Size Scaling (Power Lines)
**Concept:** Larger models can tolerate (and benefit from) larger batch sizes because their gradients are "noisier" (more variance).
*   **The Formula:** $B_{opt} \propto D^{0.383}$ (where D is the training horizon/depth).
*   **Why:** Increasing batch size improves parallelism but has diminishing returns for convergence. This power law (from "Power Lines" paper) finds the sweet spot where the extra compute of a larger batch pays for itself in faster convergence.

### 3. Learning Rate Scaling
**Concept:** If you increase the batch size, you take fewer steps. To make progress, you must take *larger* steps.
*   **The Formula:** $\text{LR} \propto \sqrt{\text{BatchSize} / \text{RefBatchSize}}$.
*   **Why:** This is a standard heuristic for AdamW. It ensures the total "distance" traveled in parameter space remains consistent regardless of batch size.

### 4. Weight Decay Scaling
**Concept:** Weight decay prevents overfitting by penalizing large weights.
*   **The Finding:** Nanochat experiments found that optimal weight decay scales inversely with the square of the model width: $\text{WD} \propto 1 / \text{Width}^2$.
*   **Why:** Wider networks have more connections. To keep the total influence on the next layer consistent, individual weights must be smaller. Stronger weight decay enforces this.

### 5. FP8 Training (H100 Optimization)
**Concept:** Modern H100 GPUs have tensor cores that run 2x faster on 8-bit floats (FP8) than 16-bit (BF16).
*   **Tensor-wise vs. Row-wise:**
    *   **Row-wise:** Each row of a matrix has its own scaling factor. More accurate, but slower (overhead of computing scales).
    *   **Tensor-wise:** The entire matrix shares one scaling factor. Slightly less accurate, but much faster.
*   **Nanochat Choice:** Uses **Tensor-wise** FP8 (`--fp8-recipe=tensorwise`). The slight precision loss is offset by the massive throughput gain, allowing us to train for more steps in the same time.

---

## Other Scripts

### `base_eval.py` (The Yardstick)
*   **Metric:** Uses **BPB (Bits Per Byte)**.
    *   **Why BPB?** Cross-entropy loss depends on vocabulary size. BPB standardizes this, making it comparable across different tokenizers.
    *   $BPB = (Loss / \ln(2)) / \text{BytesPerToken}$.
*   **CORE Metric:** A holistic benchmark aggregating performance on standard tasks (MMLU, etc.) to give a single "Capability" score.

### `chat_sft.py` (Personality Injection)
*   **Purpose:** Pretraining teaches the model "English and Facts". SFT (Supervised Fine-Tuning) teaches it "How to be an Assistant".
*   **Method:** Trains on specific "User: ... Assistant: ..." formatted data.
*   **Identity:** Includes a synthetic dataset (`identity_conversations.jsonl`) to teach the model its name ("Nanochat") and creator, reducing hallucinations about being "created by OpenAI".