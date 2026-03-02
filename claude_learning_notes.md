# Claude's Enhanced Learning Notes: nanochat/gpt.py

**Purpose**: Comprehensive deep dive into `nanochat/gpt.py` with verified technical details, exact line references, and complete code snippets.

**Based on**: nanochat master branch (verified 2026-02-22)

---

## 0. Recommended Reading Order

For deep understanding of `gpt.py`, follow this sequence:

1. **GPTConfig** dataclass (lines 28-40) - architecture parameters
2. **_precompute_rotary_embeddings** (lines 272-289) - RoPE cache setup
3. **apply_rotary_emb** (lines 51-59) - the rotation operation
4. **CausalSelfAttention.__init__** (lines 61-82) - projection matrices
5. **CausalSelfAttention.forward** (lines 84-132) - attention mechanism
6. **MLP** (lines 135-147) - feed-forward network
7. **Block** (lines 150-161) - combining attn + mlp
8. **GPT.forward** (lines 419-461) - complete forward pass
9. **init_weights** (lines 216-270) - initialization strategy
10. **setup_optimizer** (lines 379-417) - parameter grouping

---

## 1. High-Level Architecture

### The Master Dial: `--depth`

Everything scales from a single parameter. Actual calculation from `scripts/base_train.py` lines 128-130:

```python
base_dim = depth * args.aspect_ratio  # Default: depth * 64
model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim  # Round up for clean division
num_heads = model_dim // head_dim
```

**Example**: `depth=26, aspect_ratio=64, head_dim=128`
- `base_dim = 26 * 64 = 1664`
- `model_dim = ((1664 + 127) // 128) * 128 = 1664` (already multiple of 128)
- `num_heads = 1664 // 128 = 13`

### Vocab Padding Optimization (lines 176-180)

```python
padded_vocab_size = ((vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
# 32768 → 32832 (nearest multiple of 64)
```

**Why?**
- Better tensor core utilization (64-aligned)
- More efficient DDP communication
- Outputs are cropped back to real vocab size (line 450)

### Sliding Window Attention (SSSL)

**Pattern**: "SSSL" = 3 short layers, 1 long layer
- **S (Short)**: 1024 token window (half context)
- **L (Long)**: 2048 token window (full context)
- Saves ~40% computation
- Defined in `_compute_window_sizes` (lines 291-318)

**Window Size Format**: `(left, right)` tuples for Flash Attention
- `left`: tokens before current position (`-1` = unlimited)
- `right`: tokens after current position (`0` for causal)

**Examples**:
- `(2048, 0)` = Full context, causal (final layer always uses this)
- `(1024, 0)` = Short window (half context), causal
- `(-1, -1)` = Full bidirectional (not used in nanochat)

---

## 2. Meta Device Initialization Pattern

**Three-Stage Efficient Initialization** (pattern from lines 137-147 in base_train.py):

```python
# Stage 1: Build on meta device (shapes/dtypes only, no data allocation)
with torch.device("meta"):
    model = GPT(config)

# Stage 2: Allocate storage on target device (uninitialized data)
model.to_empty(device=device)

# Stage 3: Initialize weights with proper distributions
model.init_weights()
```

**Why this pattern?**
- Avoids double memory allocation for large models
- First allocation would be on CPU, then move to GPU = 2x memory
- Efficient for 1.6GB+ models (GPT-2 scale)

---

## 3. The Complete Forward Pass Flow

Tracing `GPT.forward` (lines 419-461):

### Step 1: Rotary Embeddings Setup (lines 423-431)

```python
# Grab RoPE cache for current sequence length
assert T <= self.cos.size(1), f"Sequence length grew beyond cache"

# If using KV cache (inference), offset to current position
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

**Key insight**: RoPE cache is pre-computed for `sequence_len * 10` positions, sliced as needed.

### Step 2: Token Embedding + Initial Norm (lines 434-436)

```python
x = self.transformer.wte(idx)    # Token lookup: (B, T) → (B, T, n_embd)
x = norm(x)                       # Initial RMSNorm (stabilize residual stream)
x0 = x                            # Save for skip connections
```

**Why initial norm?** Ensures starting signal has unit variance.

### Step 3: Transformer Layers (lines 437-444)

```python
for i, block in enumerate(self.transformer.h):
    # Apply per-layer scaling (learnable coefficients)
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

    # Get value embeddings if this layer has them (alternating pattern)
    ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None

    # Process through block (attention + MLP with pre-norm)
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
```

**Dual residual stream**:
- Main path: scaled by `resid_lambdas[i]` (init 1.0)
- Skip from input: scaled by `x0_lambdas[i]` (init 0.1)

### Step 4: Final Norm (line 445)

```python
x = norm(x)  # Final RMSNorm before output projection
```

### Step 5: Output Projection (lines 448-452)

```python
logits = self.lm_head(x)                          # (B, T, padded_vocab_size)
logits = logits[..., :self.config.vocab_size]    # Remove padding
logits = logits.float()                           # Switch to fp32 for stability
softcap = 15
logits = softcap * torch.tanh(logits / softcap)  # Soft-cap to [-15, 15]
```

**Soft-capping**: Prevents extreme logit values that cause training instability.

### Step 6: Loss Computation (lines 454-458)

```python
if targets is not None:
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # Flatten: (B*T, vocab)
        targets.view(-1),                   # Flatten: (B*T,)
        ignore_index=-1,                    # Skip masked tokens
        reduction=loss_reduction            # 'mean' or 'sum'
    )
    return loss
else:
    return logits  # Inference mode
```

---

## 4. The Input Stage

### Embedding Normalization (line 435)

```python
x = self.transformer.wte(idx)  # Token embedding lookup
x = norm(x)                     # Immediate RMSNorm
```

**Implementation of norm** (lines 42-44):
```python
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))
```

**Why no learnable parameters?**
- Simpler than LayerNorm (no γ, β to learn)
- Sufficient for normalizing variance
- Reduces parameter count

### The x0 Skip Connection (lines 436, 440)

```python
x0 = x  # Save normalized embedding

# Later, in each layer:
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

**Initialization** (lines 249-250):
```python
self.resid_lambdas.fill_(1.0)   # Neutral residual connection
self.x0_lambdas.fill_(0.1)      # Small skip connection weight
```

**Why 0.1 for x0_lambdas?**
- Starts with small influence
- Prevents signal decay in deep networks (26+ layers)
- ResNet-style "information highway"

---

## 5. The Attention Mechanism (`CausalSelfAttention`)

### Projection Matrices (lines 71-78)

```python
self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)      # Query (all heads)
self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)  # Key (GQA)
self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)  # Value (GQA)
self.c_proj = nn.Linear(n_embd, n_embd, bias=False)             # Output projection
```

**Group-Query Attention (GQA)**:
- `n_kv_head` ≤ `n_head` (e.g., 4 KV heads for 12 query heads)
- KV cache is 3x smaller (memory efficient inference)
- Query heads grouped and broadcast to KV heads

**No bias**: Reduces parameters, sufficient for modern architectures

### Forward Pass: Native Layout (lines 89-93)

```python
# Project input to Q, K, V with native (B, T, H, D) layout
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

**Why (B, T, H, D)?**
- Flash Attention 3's native format
- Avoids expensive `.transpose()` operations
- Standard PyTorch uses (B, H, T, D) - requires transpose

### ResFormer Value Embeddings (lines 96-101)

```python
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))  # Range (0, 2)
    v = v + gate.unsqueeze(-1) * ve
```

**Components**:
- `ve`: Learnable value embedding (per-token, per-head)
- `ve_gate`: Linear layer using first 32 channels (line 82)
- `gate`: Input-dependent scalar per head

**Initialization** (lines 258-259):
```python
torch.nn.init.zeros_(ve_gate.weight)  # Start at sigmoid(0) = 0.5 → scaled to 1.0
```

**Which layers?** (lines 47-49):
```python
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2
```
Alternating layers, with last layer always included.

**Benefits**:
- Adds parameter capacity (memory) at near-zero FLOP cost
- Value embeddings don't participate in attention computation
- Gate allows model to learn when to use them

### RoPE Application (line 106)

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

**Critical**: Only Q and K are rotated, **NOT** V!

**Implementation** (lines 51-59):
```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # Split into pairs
    y1 = x1 * cos + x2 * sin         # Rotate first half
    y2 = x1 * (-sin) + x2 * cos      # Rotate second half
    return torch.cat([y1, y2], 3)
```

**Math**: Implements 2D rotation matrix without complex numbers.

### QK Normalization (line 107)

```python
q, k = norm(q), norm(k)  # RMS normalization
```

**Why after RoPE?**
- Stabilizes attention logits (prevents explosion in deep layers)
- Doesn't interfere with rotation (rotation is applied first)
- Acts as "temperature" control for softmax

**Effect**: Dot products `q @ k` stay bounded regardless of layer depth.

### Flash Attention with KV Cache (lines 112-127)

```python
if kv_cache is None:
    # Training: causal attention with sliding window
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    # Inference: use KV cache management
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k, v=v,  # New keys/values to append
        cache_seqlens=kv_cache.cache_seqlens,
        causal=True,
        window_size=window_size,
    )
    # Final layer advances the cache position
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)
```

**KV Cache Pattern**:
1. Get pre-allocated cache buffers for this layer
2. Append new K/V to cache (in-place)
3. Compute attention over full cache
4. Last layer advances the global position pointer

### Output Projection (lines 129-131)

```python
y = y.contiguous().view(B, T, -1)  # Flatten heads: (B, T, H*D)
y = self.c_proj(y)                  # Project back: (B, T, n_embd)
return y
```

---

## 6. The MLP & Block

### MLP Structure (lines 135-147)

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)    # Expansion
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)  # Projection

    def forward(self, x):
        x = self.c_fc(x)           # Expand to 4x dimension
        x = F.relu(x).square()     # ReLU² activation
        x = self.c_proj(x)         # Project back
        return x
```

**ReLU² vs alternatives**:
- **vs ReLU**: Smooth at zero (better gradients)
- **vs GeLU**: Cheaper to compute
- **vs SwiGLU**: 2 matrix multiplies instead of 3

**Trade-off**: Simplicity and speed over theoretical performance for small models.

### Block Structure (lines 150-161)

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

**Pre-Norm Architecture**:
- Norm applied **before** sublayer (attn/mlp), not after
- Main residual stream `x` stays "clean" (unnormalized)
- Benefits: stable gradients, easier training

**Flow**:
```
x → norm → attn → add to x
  → norm → mlp → add to x
```

---

## 7. Weight Initialization Strategy

From `init_weights()` (lines 216-270):

### Embeddings (lines 232-233)

```python
torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)  # Very small!
```

**Why lm_head std=0.001?**
- Start with very small logits
- Prevents confident (wrong) predictions at initialization
- Loss starts near log(vocab_size) = uniform distribution

### Transformer Blocks: Uniform Init (lines 236-246)

```python
n_embd = self.config.n_embd
s = 3**0.5 * n_embd**-0.5  # √3/√n_embd for Uniform to match Normal std

for block in self.transformer.h:
    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
    torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
    torch.nn.init.zeros_(block.attn.c_proj.weight)      # Output projection = 0
    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
    torch.nn.init.zeros_(block.mlp.c_proj.weight)       # Output projection = 0
```

**Why Uniform instead of Normal?**
- Avoids outliers that can occur with Normal at small scales
- Uniform(−s, s) has same variance as Normal(0, s/√3)
- √3 factor adjusts for uniform distribution

**Why Output Projections = 0?**
- Residual blocks start as identity functions: `x + 0`
- Network is "born" with identity mappings
- Training stability: early gradients flow through residual path

### Per-Layer Scalars (lines 249-250)

```python
self.resid_lambdas.fill_(1.0)   # Neutral: x → 1.0 * x
self.x0_lambdas.fill_(0.1)      # Small: x → x + 0.1 * x0
```

**Evolution during training**:
- Model can learn to adjust these dynamically
- `resid_lambdas` might decrease to dampen strong signals
- `x0_lambdas` might increase to use input embedding more

### Value Embeddings (lines 252-254)

```python
for ve in self.value_embeds.values():
    torch.nn.init.uniform_(ve.weight, -s, s)
```

Same distribution as attention projections.

### Gate Weights (lines 257-259)

```python
for block in self.transformer.h:
    if block.attn.ve_gate is not None:
        torch.nn.init.zeros_(block.attn.ve_gate.weight)
```

**Result**: `gate = 2 * sigmoid(0) = 2 * 0.5 = 1.0` (neutral start)

### Rotary Embeddings (lines 262-264)

```python
head_dim = self.config.n_embd // self.config.n_head
cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
self.cos, self.sin = cos, sin
```

**Precomputation** (lines 272-289):
```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    # Frequency for each dimension pair
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Position indices
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product: (seq_len, head_dim/2)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()

    # Convert to bfloat16 and add dims for broadcasting
    cos, sin = cos.bfloat16(), sin.bfloat16()
    cos = cos[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
    sin = sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
    return cos, sin
```

**Shape**: `(1, seq_len, 1, head_dim/2)` broadcasts to `(B, T, H, D/2)`

### Embeddings to BF16 (lines 266-270)

```python
if self.transformer.wte.weight.device.type == "cuda":
    self.transformer.wte.to(dtype=torch.bfloat16)
    for ve in self.value_embeds.values():
        ve.to(dtype=torch.bfloat16)
```

**Why?** Optimizer can tolerate BF16 embeddings, saves memory.

---

## 8. Parameter Grouping for Optimizer

From `setup_optimizer` (lines 379-417):

### Parameter Separation (lines 383-390)

```python
# Group parameters by type
matrix_params = list(self.transformer.h.parameters())  # All linear layers
value_embeds_params = list(self.value_embeds.parameters())
embedding_params = list(self.transformer.wte.parameters())
lm_head_params = list(self.lm_head.parameters())
resid_params = [self.resid_lambdas]
x0_params = [self.x0_lambdas]

# Sanity check: all parameters accounted for
assert len(list(self.parameters())) == len(matrix_params + embedding_params +
                                           lm_head_params + value_embeds_params +
                                           resid_params + x0_params)
```

### Learning Rate Scaling (lines 392-394)

```python
model_dim = self.config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
print0(f"Scaling LR for AdamW params ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
```

**Example**: `model_dim=1664` → `dmodel_lr_scale = (1664/768)^(-0.5) ≈ 0.67`

### AdamW Groups (lines 397-404)

```python
param_groups = [
    # Unembedding (lm_head)
    dict(kind='adamw', params=lm_head_params,
         lr=unembedding_lr * dmodel_lr_scale,  # 0.004 * 0.67 ≈ 0.0027
         betas=adam_betas, eps=1e-10, weight_decay=0.0),

    # Embeddings (wte)
    dict(kind='adamw', params=embedding_params,
         lr=embedding_lr * dmodel_lr_scale,  # 0.2 * 0.67 ≈ 0.13
         betas=adam_betas, eps=1e-10, weight_decay=0.0),

    # Value embeddings
    dict(kind='adamw', params=value_embeds_params,
         lr=embedding_lr * dmodel_lr_scale,  # 0.2 * 0.67 ≈ 0.13
         betas=adam_betas, eps=1e-10, weight_decay=0.0),

    # Residual scaling factors
    dict(kind='adamw', params=resid_params,
         lr=scalar_lr * 0.01,  # 0.5 * 0.01 = 0.005
         betas=adam_betas, eps=1e-10, weight_decay=0.0),

    # x0 skip connection factors
    dict(kind='adamw', params=x0_params,
         lr=scalar_lr,  # 0.5
         betas=(0.96, 0.95),  # Higher beta1 for more momentum
         eps=1e-10, weight_decay=0.0),
]
```

**Default values**:
- `unembedding_lr = 0.004`
- `embedding_lr = 0.2`
- `scalar_lr = 0.5`
- `adam_betas = (0.8, 0.95)`

### Muon Groups (lines 406-411)

```python
# Group matrices by shape for efficient stacking
for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='muon',
        params=group_params,
        lr=matrix_lr * max(1.0, shape[-2] / shape[-1]) ** 0.5,
        momentum=0.95,
        ns_steps=5,  # Newton-Schulz iterations
        beta2=0.95,  # NorMuon variance reduction
        weight_decay=weight_decay,
    ))
```

**Aspect ratio scaling**:
- Tall matrices (shape[-2] > shape[-1]): Higher LR
- Wide matrices: Lower LR
- Square matrices: Base LR

**Example**: Shape (768, 3072) → `lr = 0.02 * (3072/768)^0.5 = 0.02 * 2 = 0.04`

**Why group by shape?**
- Muon stacks parameters for single fused kernel
- All params in a group must have identical shape
- Efficient GPU utilization

### Key Insight: 30x Learning Rate Advantage

```python
# Typical AdamW for matrices: lr ≈ 0.0003
# Muon for matrices: lr = 0.02

# Speedup: 0.02 / 0.0003 ≈ 67x base LR (even higher than 30x!)
```

**Why Muon can use higher LR**:
- Orthogonalization prevents gradient explosion
- Update direction is stable (bounded singular values)
- Can take larger steps without destabilizing training

---

## 9. FLOPS Estimation (lines 323-348)

```python
def estimate_flops(self):
    """
    Return estimated FLOPs per token (forward + backward).
    Each matmul param contributes 2 FLOPs (multiply, accumulate) in forward,
    4 FLOPs in backward => 6 total.
    Plus attention FLOPs: 12 * h * q * effective_seq_len per layer.
    """
    nparams = sum(p.numel() for p in self.parameters())

    # Exclude non-matmul params
    value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
    nparams_exclude = (self.transformer.wte.weight.numel() +
                      value_embeds_numel +
                      self.resid_lambdas.numel() +
                      self.x0_lambdas.numel())

    h = self.config.n_head
    q = self.config.n_embd // self.config.n_head  # head_dim
    t = self.config.sequence_len

    # Sum attention FLOPs per layer with sliding window
    attn_flops = 0
    for window_size in self.window_sizes:
        window = window_size[0]  # left context
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq

    num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
    return num_flops_per_token
```

**Breakdown**:
- **6N**: Matrix parameters (forward 2N + backward 4N)
- **12hqT**: Attention per layer (Q@K and weighted sum of V)
- **Sliding window**: Reduces T to window size for some layers

**Usage**: Calculate MFU (Model FLOPS Utilization) = actual_flops / gpu_peak_flops

---

## 10. Parameter Scaling Analysis (lines 350-377)

```python
def num_scaling_params(self):
    """
    Return parameter counts for scaling law analysis.
    Different papers use different conventions:
    - Kaplan: exclude embeddings
    - Chinchilla: include all
    """
    wte = sum(p.numel() for p in self.transformer.wte.parameters())
    value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
    lm_head = sum(p.numel() for p in self.lm_head.parameters())
    transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
    scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
    total = wte + value_embeds + lm_head + transformer_matrices + scalars

    return {
        'wte': wte,
        'value_embeds': value_embeds,
        'lm_head': lm_head,
        'transformer_matrices': transformer_matrices,
        'scalars': scalars,
        'total': total,
    }
```

**Used for**:
- Chinchilla scaling law calculations
- Training horizon determination
- Comparing model sizes across architectures

---

## Key Takeaways

### Design Principles

1. **Simplicity**: No learnable params in norms, no bias in linear layers
2. **Efficiency**: Native FA3 layout, vocab padding, sliding windows
3. **Stability**: Output projections init to 0, QK norm, logit soft-capping
4. **Flexibility**: Meta device init, alternating value embeddings, dual optimizer

### Performance Optimizations

1. **Memory**: GQA (3x smaller KV cache), vocab padding (aligned access)
2. **Compute**: Sliding windows (40% reduction), ReLU² (2 matmuls vs 3)
3. **Training**: Muon (30x higher LR), BF16 embeddings, fused kernels

### Critical Details

1. **RoPE**: Only applied to Q/K, not V
2. **QK Norm**: After RoPE, before attention
3. **Value Embeddings**: Alternating layers, gated addition
4. **x0 Skip**: Always from initial normalized embedding, not current layer
5. **Output Projection**: Always initialized to 0 for stability

---

**End of Enhanced Learning Notes for nanochat/gpt.py**

Ready for deep dive! Follow the reading order in Section 0 for optimal learning path.
