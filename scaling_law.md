# LLM Scaling Laws: The Physics of Intelligence

This document summarizes the two most critical papers that transformed LLM training from "alchemy" into "science."

## 1. The Kaplan Scaling Laws (2020)
**Paper:** *Scaling Laws for Neural Language Models* (OpenAI)  
**Link:** [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

### 🔑 Key Insight: Power Laws Everywhere
The performance of a language model (measured by Test Loss $L$) behaves predictably according to a **power law** with respect to three scale factors:
1.  **$N$**: Number of Parameters (Model Size)
2.  **$D$**: Dataset Size (Number of Tokens)
3.  **$C$**: Compute (FLOPs used for training)

The relationship is:
$$ L(N, D) \approx \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{D_c}{D} \right)^{\alpha_D} $$

Where $N_c, D_c, \alpha_N, \alpha_D$ are constants.

### 🧠 What to Learn
1.  **Performance is Predictable:** You don't need to train a massive model to know how good it will be. You can train small models, plot their loss on a log-log scale, and extrapolate the line to predict the loss of a $100M run.
2.  **Architecture Doesn't Matter (Much):** The precise shape of the model (depth vs. width, number of heads) matters far less than the raw scale ($N$). A 10-layer model and a 20-layer model with the same parameter count perform roughly the same.
3.  **Sample Efficiency:** Larger models are more sample-efficient. They reach the same level of performance as a smaller model using *fewer* training tokens.
    *   *Engineering Implication:* If you have limited data, train a bigger model.

### ⚠️ The Kaplan "Error"
Kaplan et al. concluded that you should scale model size **much faster** than data size (specifically $N \propto D^{0.74}$ or roughly a 5:1 ratio of parameters to tokens). This led to an era of "massive models, undertrained" (like GPT-3, which was 175B params trained on only 300B tokens).

---

## 2. The Chinchilla Scaling Laws (2022)
**Paper:** *Training Compute-Optimal Large Language Models* (DeepMind)  
**Link:** [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

### 🔑 Key Insight: The Compute-Optimal Frontier
Hoffmann et al. revisited Kaplan's experiments and found a flaw: Kaplan fixed the learning rate schedule, which hurt smaller models. When tuned correctly, the optimal relationship is **linear**.

To get the best model for a fixed Compute Budget $C$:
$$ N_{opt} \propto C^{0.5} $$
$$ D_{opt} \propto C^{0.5} $$

This implies that **Model Size ($N$) and Data Size ($D$) should scale equally.**

### 🧠 The "Chinchilla Ratio" (20:1)
The paper derives a simple rule of thumb for compute-optimal training:
$$ \text{Tokens} \approx 20 \times \text{Parameters} $$

*   **Example:** If you want to train a 10B parameter model, you should train it on ~200B tokens.
*   **Correction to GPT-3:** GPT-3 (175B) was severely undertrained. By Chinchilla rules, it should have been a ~67B model trained on more data, or trained on ~3.5 Trillion tokens (which they didn't have).

### 🛠️ Engineering Takeaways for `nanochat`
1.  **Don't Overbuild:** Building a massive model and starving it of data is a waste of GPU hours. A smaller model trained longer is cheaper to run (inference) and smarter.
2.  **Inference-Optimal vs. Compute-Optimal:**
    *   **Compute-Optimal (Training):** The 20:1 ratio gives you the lowest loss *for the training money*.
    *   **Inference-Optimal (Production):** In the real world, inference costs (running the model for users) dominate. Therefore, we often **"over-train"** models (e.g., Llama 3 is trained on ~15T tokens for 8B params, a ratio of **1875:1**!).
    *   *Why?* A smaller model is faster to run. Spending extra money upfront to train a small model on massive data saves millions in the long run.

---

## Summary Comparison

| Feature | Kaplan (2020) | Chinchilla (2022) | Modern Practice (Llama 3 / Nanochat) |
| :--- | :--- | :--- | :--- |
| **Scaling Focus** | Scale Parameters > Data | Scale Parameters = Data | Scale Data >>> Parameters |
| **Optimal Ratio** | $N \propto D^{0.74}$ | $N \approx 20D$ | $N \approx 100D - 1000D$ |
| **Philosophy** | "Make the brain bigger." | "Balance brain and book." | "Read the whole library." |

### Application in `nanochat`
In `scripts/base_train.py`, you will see:
```python
target_tokens = int(args.target_param_data_ratio * num_scaling_params)
```
This is a direct application of Chinchilla. We set the ratio (e.g., 10.5 or 20) and calculate exactly how long to train.
