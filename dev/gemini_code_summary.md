# `dev` Directory Documentation

This directory captures the **research process**. In Deep Learning, understanding what *didn't* work is often as valuable as understanding what did.

## `LOG.md`: The Research Notebook

This file serves as a chronological record of experiments. It is highly recommended reading.

### Key "Negative Results" (What failed?)

1.  **Multi-Token Prediction (MTP)**
    *   **Idea:** Train the model to predict the next 2-4 tokens simultaneously (using multiple heads).
    *   **Hypothesis:** Forces the model to "plan ahead" and learn better representations.
    *   **Reality:** It added parameter overhead and complexity but did not improve the validation loss per wall-clock second. The gradient signal from the 2nd/3rd token was too noisy or redundant.

2.  **Varlen (Variable Length) Attention**
    *   **Idea:** Prevent attention from crossing document boundaries within a single packed training sequence.
    *   **Hypothesis:** "Leaking" attention from Doc B into Doc A (because they are packed in the same row) confuses the model.
    *   **Reality:** The improvement was negligible (< 0.0002 BPB). The model is smart enough to learn that a BOS (Beginning of Sequence) token effectively resets the context.

3.  **SwiGLU**
    *   **Idea:** Use the SwiGLU activation function (standard in Llama models).
    *   **Reality:** While mathematically superior, it requires 3 linear projections instead of 2. For the specific parameter budget of nanochat, squared ReLU was more efficient (better loss per second).

### Key "Positive Results" (What worked?)

1.  **Muon Optimizer:** The shift from pure AdamW to Muon for matrix weights was the largest single speedup factor.
2.  **Value Embeddings:** Adding extra capacity to the Value heads proved to be a "free lunch" – improving loss without slowing down training.
3.  **FP8:** Critical for the final 2x speedup on H100s.

## `gen_synthetic_data.py`

*   **Concept:** "Identity" is not learned from the web (the web doesn't know about "Nanochat").
*   **Mechanism:** We generate synthetic dialogues where a user asks "Who are you?" and the assistant answers "I am Nanochat...".
*   **Usage:** This data is mixed into the SFT stage to prevent the model from hallucinating that it is ChatGPT or GPT-4.