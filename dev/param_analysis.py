def count_params(n_layer, n_embd, vocab_size=50257, ctx_len=1024, name="GPT-2"):
    # Embeddings
    wte = vocab_size * n_embd
    wpe = ctx_len * n_embd
    total_embed = wte + wpe

    # Transformer Block (per layer)
    # Attn: c_attn (3*d*d) + c_proj (d*d) = 4*d*d
    # MLP: c_fc (d*4d) + c_proj (4d*d) = 8*d*d
    # Biases: (3d + d) + (4d + d) = 9d
    # LayerNorms: 2 * (2d) = 4d
    params_per_layer = 12 * n_embd**2 + 13 * n_embd

    total_body = n_layer * params_per_layer

    # Final LayerNorm
    final_ln = 2 * n_embd

    total_params = total_embed + total_body + final_ln
    
    # Calculate percentages
    body_pct = (total_body / total_params) * 100
    embed_pct = (total_embed / total_params) * 100
    
    return {
        "name": name,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "total": total_params,
        "body": total_body,
        "embed": total_embed,
        "body_pct": body_pct,
        "embed_pct": embed_pct
    }

configs = [
    (12, 768, "GPT-2 Small"),
    (24, 1024, "GPT-2 Medium"),
    (36, 1280, "GPT-2 Large"),
    (48, 1600, "GPT-2 XL"),
    (96, 12288, "GPT-3 175B"),
]

print(f"{'Model':<15} {'Params':<12} {'Embeddings':<12} {'Linear Body':<12} {'Body %':<8}")
print("-" * 65)

for layers, d_model, name in configs:
    # GPT-3 uses larger ctx len usually (2048) and larger vocab sometimes, but let's stick to GPT-2 defaults for comparison or update for GPT-3
    ctx = 2048 if "GPT-3" in name else 1024
    res = count_params(layers, d_model, ctx_len=ctx, name=name)
    print(f"{res['name']:<15} {res['total']/1e6:<10.1f}M {res['embed']/1e6:<10.1f}M {res['body']/1e6:<10.1f}M {res['body_pct']:<8.1f}%")
