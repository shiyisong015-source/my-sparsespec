"""Model adapter for a batch of full-prefill and compact-draft requests.

Implements the three KV calls made by model.Attention: RoPE, append, attention.
Attach with Transformer.attach_kv_pool(adapter) for one eager model forward,
then call finish() only after the entire model forward returns. This adapter
does not implement the legacy scheduler's page-allocation/metadata contract.
Each TP rank builds identical request ordering using its own local KV heads.
"""

import torch


class PackedBatch:
    """One batched model forward, with dense projections/MLP kept batched.

CUDA attention uses one shared hot/cold Split-KV grid over the entire batch.
The torch backend remains an independent per-request mathematical reference.
The caller coordinates all TP ranks before constructing/forwarding a batch.
"""

    def __init__(self, frames, *, rope_fn=None):
        self.frames = tuple(frames)
        if not self.frames:
            raise ValueError("A packed batch needs at least one request")
        first = self.frames[0].region
        geometry = (first.device, first.dtype, first.layers, first.heads, first.dim)
        if len({id(frame.region) for frame in self.frames}) != len(self.frames):
            raise ValueError("A packed request may occur only once in a batch")
        for frame in self.frames:
            region = frame.region
            if ((region.device, region.dtype, region.layers, region.heads, region.dim) != geometry
                    or not getattr(region, "_forward_active", False)
                    or frame._layers_done or frame._failed or frame._finished):
                raise ValueError("Packed batch needs fresh frames with matching rank-local geometry")
        self.device, self.dtype, self.layers, self.heads, self.dim = geometry
        self._offsets = [0]
        for frame in self.frames:
            self._offsets.append(self._offsets[-1] + frame.query_tokens)
        self.num_tokens = self._offsets[-1]
        self._host_rope_metadata = torch.tensor(
            self._offsets + [frame.position_start for frame in self.frames],
            dtype=torch.int32, device="cpu", pin_memory=self.device.type == "cuda")
        self._rope_metadata = self._host_rope_metadata.to(self.device, non_blocking=True)
        self.qo_indptr = self._rope_metadata[:len(self._offsets)]
        self.position_offsets = self._rope_metadata[len(self._offsets):]
        self._rope_fn = rope_fn  # CPU tests can inject an independent position oracle.
        self._metadata_ready = (torch.cuda.current_stream(self.device).record_event()
                                if self.device.type == "cuda" else None)
        self._pending = None
        self._layers_done = set()
        self._finished = self._aborted = False
        self._gpu_attention = None
        self._compute_stream = None
        self._query_heads = None

    def _require_open(self):
        if self._finished or self._aborted:
            raise RuntimeError("Packed model batch is no longer writable")

    def _check_kv(self, keys, values):
        shape = (self.num_tokens, self.heads, self.dim)
        if any(tensor.shape != shape or tensor.device != self.device or tensor.dtype != self.dtype
               for tensor in (keys, values)):
            raise ValueError("K/V must match the batched rank-local query geometry")

    def _check_query(self, q):
        if (q.ndim != 3 or q.shape[0] != self.num_tokens or q.shape[-1] != self.dim
                or q.shape[1] < 1 or q.shape[1] % self.heads
                or q.device != self.device or q.dtype != self.dtype):
            raise ValueError("Q must match the batched rank-local query geometry")
        if self._query_heads is None:
            self._query_heads = q.shape[1]
        elif q.shape[1] != self._query_heads:
            raise ValueError("Query head count must remain fixed across model layers")

    def _bind_stream(self):
        if self.device.type != "cuda":
            return None
        current = torch.cuda.current_stream(self.device)
        if self._compute_stream is None:
            current.wait_event(self._metadata_ready)
            self._rope_metadata.record_stream(current)
            self._compute_stream = current
        elif current != self._compute_stream:
            raise RuntimeError("All layers of a packed batch must share one compute stream")
        return current

    def apply_batch_rope(self, q, k, config):
        """Use each frame's absolute logical offset, preserving existing RoPE settings."""
        self._require_open()
        self._check_query(q)
        self._check_kv(k, k)
        if getattr(config, "high_freq_factor", None) is not None or getattr(config, "low_freq_factor", None) is not None:
            raise ValueError("This adapter supports the same basic RoPE settings as KVPool")
        self._bind_stream()
        rope_fn = self._rope_fn
        if rope_fn is None:
            if self.device.type != "cuda":
                raise ValueError("CPU batch checks must supply an independent RoPE reference")
            import flashinfer

            rope_fn = flashinfer.apply_rope
        return rope_fn(q=q, k=k, indptr=self.qo_indptr, offsets=self.position_offsets,
                       interleave=False, rope_scale=config.scaling_factor, rope_theta=config.rope_base)

    def append_batch_kv_cache(self, k, v, layer_idx):
        """Hold this layer's K/V until batch_attn_fwd writes and consumes each frame."""
        self._require_open()
        self._check_kv(k, v)
        if type(layer_idx) is not int or not 0 <= layer_idx < self.layers:
            raise ValueError("Invalid layer")
        if self._pending is not None or layer_idx in self._layers_done:
            raise RuntimeError("Complete the pending layer once before appending another")
        self._pending = (layer_idx, k, v)

    def batch_attn_fwd(self, q, layer_idx):
        """Execute full prefill, draft, or split verify on request slices."""
        self._require_open()
        self._check_query(q)
        if self._pending is None or self._pending[0] != layer_idx:
            raise RuntimeError("Append this layer's KV before computing attention")
        _, keys, values = self._pending
        try:
            use_batch = (q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16)
                         and all(getattr(getattr(f, "attention", None), "backend",
                                         getattr(f, "backend", None)) == "flashinfer" for f in self.frames))
            if use_batch:
                from serve.attention.batched_kv_attention import BatchedKVAttention

                if any(t.stride(2) != 1 or t.stride(1) != self.dim for t in (q, keys, values)):
                    raise ValueError("Batched attention requires contiguous head/channel dimensions")
                if self._gpu_attention is None:
                    self._gpu_attention = BatchedKVAttention(self.frames, self._offsets, query_heads=q.shape[1])
                current = self._bind_stream()
                for frame in self.frames:
                    frame._prepare_batch_layer(layer_idx, current)
                output = self._gpu_attention.run(q, keys, values, layer_idx)
                for frame in self.frames:
                    frame.complete_layer(layer_idx)
            else:
                outputs = [frame.layer(layer_idx, q[start:end], keys[start:end], values[start:end])
                           for frame, start, end in zip(self.frames, self._offsets[:-1], self._offsets[1:])]
                output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        except Exception:
            self.abort()
            raise
        self._pending = None
        self._layers_done.add(layer_idx)
        return output

    def finish(self):
        """Publish all request tails after the model's final MLP/projection returns.

This is rank-local completion; TP readiness and acceptance remain explicit.
"""
        self._require_open()
        if self._pending is not None or len(self._layers_done) != self.layers:
            raise RuntimeError("Every model layer must finish before completing the packed batch")
        scores = tuple(frame.finish() for frame in self.frames)
        self._finished = True
        return scores

    def abort(self):
        """Return completion events; caller retires any partially written regions."""
        if self._finished:
            raise RuntimeError("A finished packed batch cannot be aborted")
        self._aborted = True
        self._pending = None
        return tuple(frame.ready if frame._finished else frame.abort() for frame in self.frames)
