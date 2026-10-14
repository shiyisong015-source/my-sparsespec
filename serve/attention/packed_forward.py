"""Whole-forward tail and selection boundaries for a packed KV request.

This adapter is opt-in and independent of the current serving scheduler.
It supports one draft step or rewriting the full verification query tail.
Cold KV must be restored before constructing a verify frame; no layerwise transfer overlap
is introduced here. TP ranks construct frames for their local heads.
"""

import math

import torch

from serve.attention.kv_partition import PackedKVRegion, partition_pages
from serve.attention.packed_attention import PackedAttention


class PackedForward:
    """Write and attend each layer, then publish the tail after every layer."""

    def __init__(self, region, *, mode, query_tokens, restored=None, attention=None):
        _ = region.hot_kv  # Require a completed migration.
        if mode not in ("draft", "verify"):
            raise ValueError("Packed forward mode must be draft or verify")
        if type(query_tokens) is not int or query_tokens < 1:
            raise ValueError("query_tokens must be positive")
        if mode == "draft" and query_tokens != 1:
            raise ValueError("Draft appends one query token per forward")
        if mode == "verify" and restored is None:
            raise ValueError("Restore cold history before beginning verification")
        if getattr(region, "_forward_active", False):
            raise RuntimeError("A packed forward already owns this region")
        if getattr(region, "_forward_failed", False):
            raise RuntimeError("Packed KV was partially overwritten by a failed forward")
        if getattr(region, "_needs_repack", False):
            raise RuntimeError("Reselect and repack accepted history before the next draft round")
        if getattr(region, "_verification_pending", False):
            raise RuntimeError("Commit the finished verification before starting another forward")
        if mode == "verify" and query_tokens != region.tail_length + 1:
            raise ValueError("Verification rewrites the cached draft tail plus one uncached token")
        if mode == "verify" and (
                restored.kv is None or restored.offset != 0 or restored.count != region.cold_count
                or restored.kv.ndim != 6 or restored.kv.shape[0] != region.layers
                or restored.kv.shape[1] < restored.count
                or restored.kv.shape[2:] != region.hot_kv.shape[2:]
                or restored.kv.device != region.device or restored.kv.dtype != region.dtype):
            raise ValueError("Verification requires matching full restored cold history")
        self.region, self.mode = region, mode
        self.query_tokens = query_tokens
        self.write_offset = region.tail_length if mode == "draft" else 0
        # RoPE uses full logical history, never the shorter compact hot length.
        self.position_start = region.history_length + self.write_offset
        self.tail_length = self.write_offset + query_tokens
        if self.tail_length > region.tail_capacity:
            raise ValueError("Packed speculative tail capacity is insufficient")
        self.attention = PackedAttention() if attention is None else attention
        self.restored = restored
        self._layers_done = set()
        self._finished = False
        self._failed = False
        self._aborted = False
        self._accepted = self._adopted = False
        self._epoch = getattr(region, "_forward_epoch", 0) + 1
        self._compute_stream = None
        self._prior_ready = getattr(region, "_forward_ready", None)
        self.ready = None
        self.hot_scores = self.cold_scores = None
        # Allocate score outputs once. The GPU kernel writes each layer directly,
        # avoiding a full score tensor per layer followed by another stack/copy.
        if mode == "verify":
            self.hot_scores = torch.empty(
                (region.layers, region.heads, region.hot_count + query_tokens),
                dtype=torch.float16, device=region.device,
            )
            self.cold_scores = torch.empty(
                (region.layers, region.heads, region.cold_count),
                dtype=torch.float16, device=region.device,
            )
        self._setup_ready = (torch.cuda.current_stream(region.device).record_event()
                             if region.device.type == "cuda" else None)
        region._forward_active = True
        region._verification_pending = mode == "verify"

    def prepare_layer(self, layer, q, keys, values):
        """Validate and establish dependencies before batch or reference writes."""
        if self._finished or self._failed:
            raise RuntimeError("Packed forward is no longer writable")
        if type(layer) is not int or not 0 <= layer < self.region.layers:
            raise ValueError("Invalid layer")
        if layer in self._layers_done:
            raise RuntimeError("A layer can be forwarded only once per frame")
        if (q.ndim != 3 or q.shape[0] != self.query_tokens
                or keys.ndim != 3 or keys.shape[0] != self.query_tokens
                or q.shape[1] < 1 or q.shape[1] % self.region.heads or q.shape[-1] != self.region.dim
                or keys.shape != values.shape or keys.shape[1:] != (self.region.heads, self.region.dim)
                or any(tensor.device != self.region.device or tensor.dtype != self.region.dtype
                       for tensor in (q, keys, values))):
            raise ValueError("Query and tail geometry must match the scheduled forward")
        self._prepare_batch_layer(layer, torch.cuda.current_stream(q.device)
                                  if q.device.type == "cuda" else None)

    def _prepare_batch_layer(self, layer, current):
        """Use batch-validated Q/K/V geometry while preserving frame guards."""
        if self._finished or self._failed or layer in self._layers_done:
            raise RuntimeError("Packed frame layer is no longer writable")
        if type(layer) is not int or not 0 <= layer < self.region.layers:
            raise ValueError("Invalid layer")
        if current is not None:
            if self._compute_stream is not None and current != self._compute_stream:
                raise RuntimeError("All layers of a packed frame must use one compute stream")
            if self._compute_stream is None:
                current.wait_event(self._setup_ready)
                if self._prior_ready is not None:
                    current.wait_event(self._prior_ready)
                if self.restored is not None:
                    current.wait_event(self.restored.ready)
                for tensor in (self.region.hot_kv, self.hot_scores, self.cold_scores):
                    if tensor is not None:
                        tensor.record_stream(current)
            self._compute_stream = current
        self._writes_started = True

    def complete_layer(self, layer):
        self._layers_done.add(layer)

    def layer(self, layer, q, keys, values):
        """Consume already normalized/rotated Q/K through the reference backend."""
        self.prepare_layer(layer, q, keys, values)
        # Once writes begin, failure must invalidate the frame: simply restoring
        # a length would not undo target KV that overwrote speculative draft KV.
        try:
            self.region.write_tail(layer, keys, values, self.write_offset)
            if self.mode == "draft":
                state = self.attention.draft_layer(q, self.region, layer,
                                                   tail_length=self.tail_length)
            else:
                state, _, _ = self.attention.verify_layer(
                    q, self.region, self.restored, layer, tail_length=self.tail_length,
                    hot_scores=self.hot_scores[layer], cold_scores=self.cold_scores[layer],
                )
        except Exception:
            self._failed = self.region._forward_failed = True
            self.region._forward_active = False
            self.region._verification_pending = False
            raise
        self._layers_done.add(layer)
        return state.output

    def finish(self):
        """Publish completed tail metadata, returning verify selection scores."""
        if self._failed:
            raise RuntimeError("Discard the region after a failed packed forward")
        if self._finished:
            raise RuntimeError("Packed forward has already finished")
        if len(self._layers_done) != self.region.layers:
            raise RuntimeError("Every layer must finish before publishing packed tail length")
        self.region.set_tail_length(self.tail_length)
        self.region._forward_epoch = self._epoch
        self.ready = (self._compute_stream.record_event() if self._compute_stream is not None else None)
        self.region._forward_ready = self.ready
        self.region._forward_active = False
        self._finished = True
        return self.hot_scores, self.cold_scores

    def accept(self, accepted_tail_tokens):
        """Truncate verified KV to accepted cached tokens; bonus output has no KV.

        This count includes the input anchor and accepted draft tokens. It is
        the number of tokens emitted by rejection sampling, including its
        bonus token, although that bonus token's own KV is still absent.
        After truncation the caller must reselect/repack before the next round.
        """
        if self.mode != "verify" or not self._finished or self._failed:
            raise RuntimeError("Only a finished verification can commit accepted KV")
        if self._accepted:
            raise RuntimeError("Verification has already committed accepted KV")
        if (getattr(self.region, "_forward_active", False)
                or getattr(self.region, "_forward_epoch", 0) != self._epoch):
            raise RuntimeError("This verification no longer owns the current tail")
        if type(accepted_tail_tokens) is not int or not 1 <= accepted_tail_tokens <= self.query_tokens:
            raise ValueError("Invalid accepted cached-tail count")
        self.region.set_tail_length(accepted_tail_tokens)
        self.region._needs_repack = True
        self.region._verification_pending = False
        self._accepted = True
        return self.region.history_length + accepted_tail_tokens

    def abort(self):
        """Cancel the frame; partial writes require retiring the entire region.

        Return the local CUDA completion event for explicit buffer retirement.
        Aborting before any layer writes leaves the prior history usable.
        """
        if self._finished:
            raise RuntimeError("A finished packed forward cannot be aborted")
        if self._aborted:
            return self.ready
        self._failed = True
        self._aborted = True
        self.region._forward_active = self.region._verification_pending = False
        if self._layers_done or getattr(self, "_writes_started", False):
            self.region._forward_failed = True
        self.ready = (self._compute_stream.record_event() if self._compute_stream is not None else None)
        return self.ready

    def selection_plan(self, *, budget_ratio, min_tokens=128, validate=False):
        """Choose the next history budget using scores and temporary segment slots.

        Preserve the most recent old-history token followed by all accepted
        tail KV. Older history can be permuted because its keys already carry
        RoPE and every query sees that entire prefix. Original token IDs are
        neither restored nor retained.
        """
        if not self._accepted or self._adopted or self.region.phase != "packed":
            raise RuntimeError("Selection requires this frame's committed verification")
        if (not isinstance(budget_ratio, (int, float)) or not math.isfinite(budget_ratio)
                or not 0 < budget_ratio <= 1 or type(min_tokens) is not int or min_tokens < 1):
            raise ValueError("Invalid next-round KV budget")
        if self.region.recent_count < 1:
            raise ValueError("Keep the latest history token in hot KV to preserve the recent suffix")
        if self.region.device.type == "cuda":
            torch.cuda.current_stream(self.region.device).wait_event(self.ready)
        hot_history = self.region.hot_count
        hot_live = self.region.hot_length
        cold_count = self.region.cold_count
        device = self.region.device
        # Candidate prefix: all old history except its latest token. Move that
        # latest token behind cold, so it stays adjacent to the accepted tail.
        prefix_slots = torch.cat((torch.arange(hot_history - 1, device=device),
                                  hot_live + torch.arange(cold_count, device=device)))
        recent_slots = torch.arange(hot_history - 1, hot_live, device=device)
        full_slots = torch.cat((prefix_slots, recent_slots)).int()
        scores = torch.cat((self.hot_scores[..., :hot_history - 1], self.cold_scores), dim=-1)
        recent = self.region.tail_length + 1
        length = self.region.history_length + self.region.tail_length
        budget = min(length, max(min_tokens, math.ceil(length * budget_ratio), recent))
        selected_count = budget - recent
        if selected_count:
            positions = torch.topk(scores, selected_count, dim=-1, sorted=True).indices
            selected = prefix_slots[positions].int()
        else:
            selected = torch.empty((self.region.layers, self.region.heads, 0),
                                    dtype=torch.int32, device=device)
        return partition_pages(full_slots, selected, selected_count, pool_capacity=length,
                               recent_count=recent, validate=validate)

    def start_repack(self, *, budget_ratio, min_tokens=128, cpu_pool=None, gpu_pool=None,
                     stream=None, chunk_pages=1024, validate=False, hot_capacity=None, cold_capacity=None):
        """Start the local migration after all layers and rejection sampling."""
        if getattr(self, "_replacement", None) is not None:
            raise RuntimeError("This verification already started a replacement migration")
        plan = self.selection_plan(budget_ratio=budget_ratio, min_tokens=min_tokens,
                                   validate=validate)
        replacement = PackedKVRegion.start_repartition(
            self.region, self.restored, plan, cpu_pool=cpu_pool, gpu_pool=gpu_pool,
            stream=stream, chunk_pages=chunk_pages, source_ready=self.ready,
            hot_capacity=hot_capacity, cold_capacity=cold_capacity,
        )
        self._replacement = replacement
        return replacement

    def adopt(self, replacement):
        """Retire old buffers after migration and caller-coordinated TP readiness.

        The caller must finish migration on every TP rank before adopting or
        forwarding the next round. This local method performs no collective.
        """
        if (not self._accepted or self._adopted
                or replacement is not getattr(self, "_replacement", None)
                or replacement.phase != "packed"
                or replacement._previous_region is not self.region):
            raise RuntimeError("Finish this frame's replacement migration before adoption")
        self.region.close(consumer_done=replacement.offload_ready)
        if self.restored._lease is not None:
            self.restored.release(consumer_done=replacement.offload_ready)
        else:
            self.restored.kv = None  # Borrowed, standalone workspace; all copies have finished.
        replacement._previous_region = replacement._restored_source = None
        self.hot_scores = self.cold_scores = None
        self._adopted = True
        return replacement
