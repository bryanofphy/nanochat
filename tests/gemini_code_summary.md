# `tests` Directory Documentation

This directory ensures the optimizations don't break the math.

## `test_engine.py`: Correctness Verification

*   **The Problem:** We implemented a custom, highly optimized `Engine` with KV-Caching in `engine.py`.
*   **The Risk:** It's easy to introduce "off-by-one" errors in KV-caching (e.g., attending to the wrong past token) which silently ruins generation quality.
*   **The Test:**
    1.  Run the **Reference Model** (slow, standard PyTorch `generate`).
    2.  Run the **Optimized Engine** (fast, custom KV cache).
    3.  Assert that the output logits/tokens match **exactly**.

## `test_attention_fallback.py`

*   **The Problem:** We support both Flash Attention 3 (Hopper) and SDPA (older GPUs).
*   **The Test:** Runs the same input through both implementations and verifies the outputs match. This ensures that users on older hardware (or MacBooks) get the same model behavior as H100 users, just slower.