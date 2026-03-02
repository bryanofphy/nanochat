# Summary: Value Residual Learning (ResFormer)

**Paper**: "Value Residual Learning For Alleviating Attention Concentration In Transformers"
**ArXiv**: https://arxiv.org/abs/2410.17897
**Authors**: Zhanchao Zhou et al., 2024

---

## Core Idea

Standard transformer residual connections only flow through the **hidden state** (residual stream). The paper argues this causes "attention concentration" in deeper layers — later layers lose access to the initial token-level information.

ResFormer adds a second residual connection directly on the **value vectors**, connecting layer 1's values to all subsequent layers.

---

## ResFormer Formula

```
U_n = (1/2) * A_n * (V_n + V_1)    for n >= 2
```

- `A_n` = attention matrix at layer n
- `V_n` = current layer's value projection
- `V_1` = first layer's value projection (fixed shortcut)
- `1/2` = fixed constant, NOT learned

**No gating mechanism.** No learnable parameters. No per-head scalars. Just a direct addition with a fixed 0.5 coefficient.

---

## SVFormer Variant (KV Cache Optimization)

```
U_n = A_n * V_1    for all n >= 2
```

All layers share the first layer's values exclusively. Reduces KV cache by ~50% since only layer 1's KV needs to be stored.

---

## What nanochat Does Differently

nanochat's implementation diverges significantly from the paper:

| Aspect | ResFormer (paper) | nanochat |
|---|---|---|
| Value source | First layer's V (reused) | Fresh token embedding lookup per layer |
| Mixing coefficient | Fixed 1/2 | Learned gate per head per token |
| Gate input | N/A | First 32 channels of residual stream |
| Gate output | N/A | One scalar per KV head, range (0,2) |
| Gate init | N/A | Zero → sigmoid(0)*2 = 1.0 (neutral) |
| Which layers | All layers except first | Alternating layers (every other) |
| Extra params | None | One full embedding table per ~2 layers |

**nanochat's `value_embeds` is a fresh embedding lookup from the vocab** — not reusing layer 1's V. This is a more expensive but more expressive variant: each layer gets a fresh, context-free view of the token's identity, rather than a recycled projection from layer 1.

**The 32-channel gate is nanochat's own addition** — not from the paper. No research paper specifically justifies using 32 channels. It is an engineering choice: the gate only needs to make a simple scalar decision per head, so a small projection (32 dims → n_kv_head) is sufficient without using the full n_embd.

---

## Relevant nanochat Code

`nanochat/gpt.py` — `CausalSelfAttention.forward()`:
```python
ve = self.value_embeds[str(i)](idx)              # (B, T, n_kv_head*head_dim) — vocab lookup
ve = ve.view(B, T, self.n_kv_head, self.head_dim)
gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))  # (B, T, n_kv_head)
v = v + gate.unsqueeze(-1) * ve
```

`nanochat/gpt.py` — `GPT.__init__()`:
```python
# Only alternating layers get value_embeds
self.value_embeds = nn.ModuleDict({
    str(i): nn.Embedding(padded_vocab_size, kv_dim)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})
```

---

## Key Takeaways

1. **ResFormer paper itself is very simple** — fixed 0.5 coefficient, no gate, reuses layer 1's V
2. **nanochat's implementation is inspired by ResFormer but substantially different** — fresh vocab lookup + learned gate is nanochat's own design
3. **The 32-channel gate has no paper citation** — it's an empirical engineering choice
4. **SVFormer** (paper variant) could be interesting for nanochat inference: skip storing KV for all layers except layer 1, saving ~50% KV cache memory
