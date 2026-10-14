"""Compact KV ownership on the original page pool; native Pillar computation.

The page table is a phase view, never an ownership list. A physical slot may
contain different historical tokens for each layer/head. Only the chronological
recent suffix and newly appended tail keep their common token order.
"""

import array
import math
from dataclasses import dataclass

import torch

from serve.attention.kv_pack import gather_to_pool, pack_selected_kv, restore_to_pool
from serve.attention.kv_partition import partition_pages
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request


@dataclass
class PageTransfer:
    pages: tuple
    ready: object
    cold: object = None
    keepalive: object = None

    def wait(self):
        self.ready.synchronize()


class CompactCachePtr(PillarCachePtr):
    """Reuse native wrappers/TopK, but own/release every physical page once.

    len_kv_cache: full logical KV history (RoPE position).
    len(indices): current append page-table length (physical append position).
    len_loading_indices: draft-visible hot + recent/tail length.

    Whole-forward callers wait on transfer events across all TP ranks before
    commit. No method silently uses the legacy Full/Streaming CPU offloader.
    """

    def __init__(self, kv_pool, indices, len_kv_cache):
        if kv_pool.page_size != 1 or len(indices) != len_kv_cache:
            raise ValueError("Compact adoption requires a complete page-size-one cache")
        self.kv_pool, self.page_size = kv_pool, 1
        self.indices = array.array("i", indices)
        self._owned = set(indices)
        if len(self._owned) != len(indices):
            raise ValueError("Cache pages must have a unique owner")
        self.indices_cpu = array.array("i")  # legacy scheduler inspection only
        self.len_kv_cache = len_kv_cache
        self.len_loading_indices = len_kv_cache
        self.len_selected_indices_gpu = self.len_selected_indices_cpu = 0
        self.selected_indices_buffer_gpu = torch.empty(
            (self.NUM_LAYERS, self.NUM_KV_HEADS, self.MAX_LEN_SELECTED_KV_CACHE),
            dtype=torch.int32, device=kv_pool.device,
        )
        self.selected_indices_ptr = self.selected_indices_buffer_gpu.data_ptr()
        self.phase = "full"
        self.hot_pages = ()
        self.restored_pages = ()
        self.tail_pages = []
        self.cold_cpu = None
        self.recent_count = 0
        self.epoch = 0
        self.verify_pages = None
        self.verify_epoch = None
        self.needs_pack = False
        self._pending = None
        self._closed = False

    @property
    def cold_count(self):
        return 0 if self.cold_cpu is None else self.cold_cpu.shape[1]

    @property
    def owned_pages(self):
        return frozenset(self._owned)

    @classmethod
    def from_kv_cache(cls, source_cache, **config):
        result = cls(source_cache.kv_pool, source_cache.indices, source_cache.len_kv_cache)
        # FullCachePtr destructor no longer owns the adopted pages.
        source_cache.indices = array.array("i")
        source_cache.len_kv_cache = 0
        return result, {"ptr": result.selected_indices_ptr, "num_selected_tokens": 0}

    def _allocate(self, count):
        if count < 0 or count > self.kv_pool.num_free_pages:
            raise MemoryError(f"Need {count} original pool pages; free={self.kv_pool.num_free_pages}")
        pages = tuple(self.kv_pool.alloc_page() for _ in range(count))
        self._owned.update(pages)
        return pages

    def _release(self, pages):
        for page in pages:
            if page not in self._owned:
                raise RuntimeError("Attempt to release a borrowed/already freed page")
            self.kv_pool.free_page(page)
            self._owned.remove(page)

    def allocate_tokens(self, num_tokens, restore=False):
        if self._pending is not None or self.needs_pack or restore or num_tokens != 1:
            raise RuntimeError("Compact append requires one token at a completed phase boundary")
        pages = self._allocate(num_tokens)
        self.tail_pages.extend(pages)
        self.indices.extend(pages)
        self.len_kv_cache += num_tokens
        self.len_loading_indices += num_tokens
        return num_tokens

    def freeze_verify_view(self):
        if self.phase not in ("full", "verify") or len(self.indices) != self.len_kv_cache:
            raise RuntimeError("Verify requires all cold restored before metadata capture")
        self.verify_pages = tuple(self.indices)
        self.verify_epoch = self.epoch

    def retract_tokens(self, num_tokens):
        if (self.verify_pages is None or self.verify_epoch != self.epoch
                or not 0 <= num_tokens <= len(self.tail_pages)):
            raise RuntimeError("Rejection may only retract this verify's chronological tail")
        if num_tokens:
            rejected = self.tail_pages[-num_tokens:]
            del self.tail_pages[-num_tokens:]
            del self.indices[-num_tokens:]
            self._release(rejected)
            self.len_kv_cache -= num_tokens
            self.len_loading_indices -= num_tokens
        if tuple(self.indices) != self.verify_pages[:self.len_kv_cache]:
            raise RuntimeError("Score page order changed before selection")

    def get_select_metadata(self, **config):
        if self.verify_pages is None or self.verify_epoch != self.epoch:
            raise RuntimeError("Selection requires the frozen current verify epoch")
        recent = min(self.len_kv_cache, config["num_accepted_tokens"] + 1)
        budget = max(config.get("num_min_budget", 128),
                     math.ceil(self.len_kv_cache * config.get("budget_ratio", .05)))
        if budget >= self.MAX_LEN_SELECTED_KV_CACHE or min(budget, self.len_kv_cache) < recent:
            raise ValueError("Hot budget must cover the forced recent suffix and be <6144")
        result = super().get_select_metadata(**config)
        self.recent_count = recent
        self.needs_pack = True
        return result

    def begin_restore(self):
        if self.phase != "hot" or self._pending is not None or self.needs_pack:
            raise RuntimeError("Restore requires a committed hot cache")
        pages = self._allocate(self.cold_count)
        try:
            if self.cold_count:
                restore_to_pool(self.cold_cpu, self.kv_pool._physical_mem_gpu, pages)
            transfer = PageTransfer(pages, torch.cuda.current_stream(self.kv_pool.device).record_event(),
                                    keepalive=self.cold_cpu)
        except Exception:
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
            self._release(pages)
            raise
        self._pending = transfer
        return transfer

    def commit_restore(self, transfer):
        if transfer is not self._pending:
            raise RuntimeError("Wrong restore transaction")
        transfer.wait()
        selected = len(self.hot_pages) - self.recent_count
        self.restored_pages = transfer.pages
        self.indices = array.array("i", self.hot_pages[:selected] + transfer.pages
                                   + self.hot_pages[selected:] + tuple(self.tail_pages))
        if len(self.indices) != self.len_kv_cache:
            raise RuntimeError("Restored page table does not cover logical history")
        self.phase, self._pending = "verify", None

    def begin_pack(self, *, validate=False):
        if (not self.needs_pack or self._pending is not None
                or self.verify_epoch != self.epoch
                or tuple(self.indices) != self.verify_pages[:self.len_kv_cache]):
            raise RuntimeError("Pack requires completed rejection/selection on the frozen verify view")
        physical = self.kv_pool._physical_mem_gpu
        full = torch.tensor(tuple(self.indices), dtype=torch.int32, device=physical.device)
        plan = partition_pages(full, self.selected_indices_buffer_gpu,
                               self.len_selected_indices_gpu, pool_capacity=self.kv_pool.capacity,
                               recent_count=self.recent_count, validate=validate)
        pages = self._allocate(plan.hot_count)
        try:
            cold = torch.empty((physical.shape[0], plan.cold_count, *physical.shape[2:]),
                               dtype=physical.dtype, device="cpu", pin_memory=True)
            gather_to_pool(physical, plan.hot_pages, pages, source_page_ids=tuple(self.indices))
            pack_selected_kv(physical, plan.cold_pages, cold, plan.cold_count)
            transfer = PageTransfer(pages, torch.cuda.current_stream(physical.device).record_event(),
                                    cold=cold, keepalive=plan)
        except Exception:
            torch.cuda.current_stream(physical.device).synchronize()
            self._release(pages)
            raise
        self._pending = transfer
        return transfer

    def commit_pack(self, transfer):
        if transfer is not self._pending:
            raise RuntimeError("Wrong pack transaction")
        transfer.wait()
        self._release(tuple(self.indices))
        self.hot_pages, self.cold_cpu = transfer.pages, transfer.cold
        self.indices = array.array("i", self.hot_pages)
        self.restored_pages, self.tail_pages = (), []
        self.len_loading_indices = len(self.hot_pages)
        count = self.len_selected_indices_gpu
        if count:
            # Future draft reads the destination slots, never the old source IDs.
            destinations = torch.tensor(self.hot_pages[:count], device=self.kv_pool.device, dtype=torch.int32)
            self.selected_indices_buffer_gpu[..., :count].copy_(destinations)
        self.phase, self.needs_pack, self._pending = "hot", False, None
        self.verify_pages = self.verify_epoch = None
        self.epoch += 1

    def abort_transfer(self, transfer):
        transfer.wait()
        self._release(transfer.pages)
        if self._pending is transfer:
            self._pending = None

    def close(self):
        if getattr(self, "_closed", True):
            return
        if self._pending is not None:
            self._pending.wait()
        self._release(tuple(self._owned))
        self.indices = array.array("i")
        self.cold_cpu = self._pending = None
        self._closed = True

    def __del__(self):
        self.close()

    def offload(self, *args, **kwargs):
        raise RuntimeError("Compact KV uses whole-forward pack transactions")

    def restore(self, *args, **kwargs):
        raise RuntimeError("Compact KV uses whole-forward restore transactions")

    @classmethod
    def dispatch_prepare_fwd_metadata(cls, **kwargs):
        host, tensors = kwargs["input_metadata"], kwargs["input_metadata_tensors"]
        requests = kwargs["active_requests"]
        # Scheduler's logical offsets remain unchanged for RoPE. The append
        # kernel separately needs positions within the physical phase page table.
        physical_lengths = array.array("i", (len(r.kv_cache_ptr.indices) for r in requests))
        host["all_kv_len_arr"] = physical_lengths
        tensors["all_kv_len_arr"] = torch.tensor(physical_lengths, dtype=torch.int32, device=kwargs["device"])
        draft_total = sum(r.len_loading_kv_cache for r in requests if r.status == Request.ReqExecType.DRAFT)
        verify_total = sum(r.len_loading_kv_cache for r in requests if r.status == Request.ReqExecType.VERIFY)
        if draft_total >= cls.MAX_TOTAL_DRAFT_KV_LEN or verify_total >= cls.MAX_TOTAL_VERIFY_KV_LEN:
            raise ValueError("Native draft/verify batch metadata capacity exceeded")
        if len(requests) >= cls._MAX_BATCH_SIZE or len(host["token_ids"]) >= cls._MAX_NNZ:
            raise ValueError("Native batch/query metadata capacity exceeded")
        for i, request in enumerate(requests):
            if request.status == Request.ReqExecType.VERIFY:
                if ((host["token_indptr"][i + 1] - host["token_indptr"][i])
                        * cls.NUM_QO_HEADS // cls.NUM_KV_HEADS > 64):
                    raise ValueError("Native score kernel supports at most 64 packed QO rows")
                request.kv_cache_ptr.freeze_verify_view()
        super().dispatch_prepare_fwd_metadata(**kwargs)
