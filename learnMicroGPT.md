# Learn MicroGPT: Deconstructing the "Ghost in the Machine"

**Date:** Feb 11, 2026  
**Source:** [Karpathy's microgpt.py](./microgpt.py)  
**Goal:** Understand the mechanical heart of Large Language Models (LLMs) without the noise of complex libraries like PyTorch or TensorFlow.

---

## 1. The Philosophy: "Everything else is just efficiency"

The most important insight from this code is that **LLMs are just math**. There is no magic neural network library handling things behind the scenes. 
*   **PyTorch** handles *parallelism* (doing math on arrays).
*   **MicroGPT** handles *logic* (doing math on scalars).

By removing the parallelism (Matrices), we expose the logic (Loops).

---

## 2. The Engine: `class Value` (Autograd)

At its core, deep learning is about minimizing a **Loss** function `L(theta)` with respect to parameters `theta`. To do this via Gradient Descent, we need to compute the gradient.

`microgpt.py` implements **Reverse-Mode Automatic Differentiation** (Autograd) via the `Value` class.

### 2.1 The Computational Graph
Each `Value` object represents a node `v` in a Directed Acyclic Graph (DAG).
*   **Primals (`v.data`):** The scalar value computed during the forward pass.
*   **Adjoints (`v.grad`):** The partial derivative `dL/dv` (how much Loss changes when `v` changes) computed during the backward pass.
*   **Edges (`v._children`):** Directed edges pointing to the operands that created `v`.

### 2.2 The Chain Rule (Backward Pass)
The `backward()` method implements the **Multivariable Chain Rule**.

For a node `v` that is an input to functions `f1`, `f2`, ... (meaning `v` is used by multiple children `c1`, `c2`, ...), the total derivative is the sum of partials from all paths.

Mathematically:
```
dL      dL     dc1      dL     dc2
--  =  --- *  ---  +   --- *  ---  + ...
dv     dc1     dv      dc2     dv
```
*(Total gradient = Sum of (Gradient from Child * Local Derivative of Child))*

### 2.3 Implementation Details

1.  **Topological Sort:**
    The graph is traversed to produce a linear ordering. We process a node only *after* all its consumers (parents in the graph) have been processed.

2.  **Gradient Accumulation (`+=`):**
    The code loop directly implements the summation:
    ```python
    child.grad += v.grad * local_grad
    ```
    *   `v.grad`: `dL/dv` (Gradient flowing from the output).
    *   `local_grad`: `dv/dchild` (Local derivative of the operation).
    *   `+=`: Ensures gradients from all branching paths are summed.

    #### Example: Multiplication
    For the operation `z = x * y`:
    *   Forward: `z.data = x.data * y.data`
    *   Local Derivatives: `dz/dx = y`, `dz/dy = x`
    *   Backward (for `x`):
        ```
        x.grad += z.grad * (dz/dx)
        x.grad += z.grad * y.data
        ```

---

## 3. The "KV Cache" Explained (Implicitly)

One of the most confusing topics in modern LLMs is the **Key-Value (KV) Cache**. In PyTorch implementations, this is often hidden or complex. In MicroGPT, it is explicitly just a Python list.

Look at the training loop:
```python
# Forward the token sequence one by one
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
for pos_id in range(n):
    # ...
    logits = gpt(token_id, pos_id, keys, values)
```

And inside `gpt()`:
```python
# Inside the attention block loop
keys[li].append(k)   # Save this token's Key
values[li].append(v) # Save this token's Value
```

### Why is this profound?
1.  **Causal Masking is Automatic:** Because we process tokens sequentially (`for pos_id in range(n)`), when we are at step 5, the lists `keys` and `values` **only contain items 0, 1, 2, 3, 4**. The model literally *cannot* see the future (token 6) because it hasn't been appended to the list yet!
2.  **The KV Cache:** The lists `keys` and `values` *are* the KV Cache. We don't recompute keys/values for previous tokens; we just append the new one and attend to the full list.

**In PyTorch:** We usually process all tokens in parallel using a triangular mask matrix (Wait O(N^2)).  
**In MicroGPT:** We process recurrently (O(N) steps, each taking O(N) time -> Total O(N^2)).

---

## 4. The Architecture Details

### Normalization: `rmsnorm`
```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```
Standard LayerNorm subtracts the mean and divides by variance. **RMSNorm** (Root Mean Square Norm) skips subtracting the mean. It just scales the vector so it has unit length. This is standard in Llama, Gemma, and modern GPTs because it's slightly faster and works just as well.

### Activation: `ReLU^2`
```python
x = [xi.relu() ** 2 for xi in x]
```
Standard GPT-2 used `GeLU` (Gaussian Error Linear Unit). Modern models (like Google's Gemma) often use `GeGLU`. Here, Karpathy uses **Squared ReLU**. It gives the non-linearity of ReLU but is smooth (differentiable) at 0, similar to GeLU, but much cheaper to compute (no `tanh` or `exp` needed).

---

## 5. The Optimizer: Adam

You often import Adam from a library. Here, you see its guts.

```python
# m: First moment (Velocity)
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
# v: Second moment (Friction/Energy)
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
# Update
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```
Adam is essentially: "Go in the direction of the gradient (`m`), but slow down if the terrain is very steep/noisy (`v`)."

---

## 6. How to Study This Code

1.  **Trace the Shapes:** Even though they are lists, imagine the shapes.
    *   `state_dict['wte']`: List of length `vocab_size` (66). Each element is list of `n_embd` (16).
    *   `q`: List of `n_embd` (16).
2.  **Print the Gradients:**
    *   After `loss.backward()`, print `params[0].grad`. See that it's non-zero.
3.  **Break the Logic:**
    *   Try removing `keys[li].append(k)`. The model will crash or fail to learn because it has no memory of the past.
    *   Try changing `n_head` to 1. Does it still learn?

## Summary

`microgpt.py` teaches us that an LLM is:
1.  **Embed:** Turn ID into vector.
2.  **Attend:** Compare current vector (Query) with past vectors (Keys) to get weights, then sum past vectors (Values).
3.  **Think:** Run result through a small neural net (MLP).
4.  **Repeat:** Do this N times (Layers).
5.  **Predict:** Turn vector back into probabilities.

All implemented with `+`, `*`, and `pow`. No magic.
