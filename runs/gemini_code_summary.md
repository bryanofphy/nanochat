# `runs` Directory Documentation

This directory contains the "recipes" for specific training outcomes. These scripts are essentially configuration files for `base_train.py`.

## `speedrun.sh`: The $100 GPT-2 Benchmark

This script represents the culmination of all optimizations in the repo. Its goal: Train a model with GPT-2 (1.5B) capabilities in ~3 hours on 8xH100s.

### Breakdown of the Recipe

1.  **Hardware**: `8xH100`.
    *   **Why:** We need massive FP8 throughput. The H100 is currently the only GPU with efficient FP8 tensor cores.
2.  **Model Depth**: `--depth=26`.
    *   **Why:** GPT-2 (Large/XL) had roughly this depth. In nanochat, depth is the "master dial" that controls all other dimensions (width, heads) via aspect ratio constraints.
3.  **Optimization**: `--fp8`.
    *   **Why:** FP8 reduces memory bandwidth usage (moving 1 byte instead of 2) and doubles compute throughput. This is the single biggest factor in dropping the cost from ~$300 to ~$72.
4.  **Data Ratio**: `--target-param-data-ratio=8.25`.
    *   **Why:** Standard Chinchilla scaling suggests ~20. However, "over-training" a smaller model (using more data than optimal) is often cheaper at inference time. Nanochat uses ~8.25-10.5 depending on the exact run, finding a sweet spot where the model is highly capable but training fits in the 3-hour window.

## `miniseries.sh`: The "Sweeper"

This script is designed to run a **Miniseries** of experiments—training multiple models of increasing depth (e.g., d12, d14, ... d26) to collect data for scaling laws or validation.

*   **Purpose:** To systematically verify that architectural changes work across *all* scales, not just for small or large models.
*   **Workflow:**
    *   Iterates through depths `[12, 14, 16, 18, 20, 22, 24, 26]`.
    *   Automatically adjusts `--device-batch-size` to prevent Out-Of-Memory (OOM) errors as models get larger.
    *   Logs key metrics (CORE score, Val BPB, Training Time) to a CSV file (`results.csv`) for easy plotting.

### `scaling_laws.sh`: The Research Lab

This script runs a sweep of smaller models (d12, d16, d20) to generate the data points used to derive the formulas in `base_train.py`.

*   **Workflow:**
    1.  Train d12, d16, d20 models.
    2.  Measure their final loss and optimal batch sizes.
    3.  Fit power-law curves ($y = ax^b$) to these points.
    4.  Hardcode the resulting exponents (like 0.383) into `base_train.py`.

### `runcpu.sh`: The Local Playground

A minimal script for debugging.
*   **Purpose:** To verify the code runs without crashing on a MacBook (MPS) or CPU.
*   **Configuration:** Drastically reduces model size (depth=3) and batch size so it runs on a laptop. **Do not expect intelligence from this model.**
