# `tasks` Directory Documentation

This directory defines the **Benchmarks** used to score the model. A raw loss number (BPB) is useful for training, but benchmarks measure actual utility.

## The Benchmark Suite

### 1. `mmlu.py` (Massive Multitask Language Understanding)
*   **What:** Multiple-choice questions covering 57 subjects (math, history, law, medicine).
*   **Why:** Measures "World Knowledge". Can the model recall facts?

### 2. `gsm8k.py` (Grade School Math 8K)
*   **What:** Multi-step math word problems.
*   **Why:** Measures "Reasoning Chain". The model must output a sequence of thought steps to reach the answer, not just a fact retrieval.

### 3. `humaneval.py`
*   **What:** Python coding problems (write a function to do X).
*   **Why:** Measures "Syntax and Logic".

### 4. `spellingbee.py`
*   **What:** Tasks like "Spell the word 'elephant'" or "How many 'p's are in 'apple'?".
*   **Why:** Tokenizers break words into chunks (e.g., "apple" -> "ap", "ple"). LLMs often struggle to "see" individual letters. This benchmark tests the model's ability to reason about the sub-token characters.

## `common.py`: The Evaluator Logic
*   **TaskMixture:** Allows combining multiple tasks into a single evaluation run.
*   **Metrics:** Defines how to score a result. Exact match? Regex match? Probability of the correct token (A/B/C/D)?