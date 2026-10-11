"""Ideal KV-data footprints from one instantaneous logical index selection."""
import numpy as np


def summarize_selection(logical_indices, context_tokens, head_dim, element_bytes):
    indices = np.asarray(logical_indices)
    if indices.ndim != 3 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Expected integer [layers, KV heads, selected tokens]")
    layers, heads, _ = indices.shape
    if min(layers, heads, context_tokens, head_dim, element_bytes) <= 0:
        raise ValueError("Dimensions must be positive")
    if indices.size and (indices.min() < 0 or indices.max() >= context_tokens):
        raise ValueError("Selection outside the request context")
    per_head = [[int(len(np.unique(indices[layer, head]))) for head in range(heads)]
                for layer in range(layers)]
    per_layer = [int(len(np.unique(indices[layer]))) for layer in range(layers)]
    union_tokens = int(len(np.unique(indices)))
    cell_bytes = 2 * head_dim * element_bytes  # K and V for one layer/head/token.
    full_bytes = layers * heads * context_tokens * cell_bytes
    retained = dict(token=union_tokens * layers * heads * cell_bytes,
                    layer=sum(per_layer) * heads * cell_bytes,
                    head=sum(map(sum, per_head)) * cell_bytes)
    if not 0 <= retained["head"] <= retained["layer"] <= retained["token"] <= full_bytes:
        raise ValueError("Invalid footprint ordering")
    return dict(context_tokens=context_tokens, layers=layers, kv_heads=heads,
                full_kv_bytes=full_bytes, union_tokens=union_tokens,
                layer_union_tokens=per_layer, per_head_unique_tokens=per_head,
                retained_kv_bytes=retained,
                releasable_kv_bytes={name: full_bytes - value for name, value in retained.items()},
                releasable_fraction={name: 1 - value / full_bytes for name, value in retained.items()},
                interpretation="Ideal KV-data capacity only; excludes allocator, workspace, packing and recovery costs")
