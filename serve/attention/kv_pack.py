"""Pack rank-local selected KV into a separate region with the original layout."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count", "destination_offset", "sl", "il", "ih", "dl", "cl", "HOT_LENGTH"])
def _pack_selected_kv(
    source, cold_source, selected, destination,
    count, destination_offset,
    layers: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr,
    sl, sp: tl.constexpr, skv: tl.constexpr,
    sh: tl.constexpr, sd: tl.constexpr,
    il, ih, ij: tl.constexpr,
    dl, dp: tl.constexpr, dkv: tl.constexpr,
    dh: tl.constexpr, dd: tl.constexpr,
    cl, cp: tl.constexpr, ckv: tl.constexpr,
    ch: tl.constexpr, cd: tl.constexpr,
    TWO_SOURCES: tl.constexpr, HOT_LENGTH,
    TOKENS: tl.constexpr, WIDTH: tl.constexpr,
):
    tiles = tl.cdiv(count, TOKENS)
    total_tasks = layers * heads * tiles
    lane = tl.arange(0, WIDTH)
    kv = lane // dim
    channel = lane % dim

    # The whole grid has only the requested number of programs. Layer/head
    # tasks are processed inside this loop, not additional grid dimensions.
    for task in range(tl.program_id(0), total_tasks, tl.num_programs(0)):
        layer_head = task // tiles
        layer = (layer_head // heads).to(tl.int64)
        head = (layer_head % heads).to(tl.int64)
        j = (task % tiles) * TOKENS + tl.arange(0, TOKENS)
        source_page = tl.load(
            selected + layer * il + head * ih + j * ij,
            mask=j < count, other=0,
        ).to(tl.int64)
        mask = (j[:, None] < count) & (lane[None, :] < 2 * dim)
        source_address = (
            source + layer * sl + source_page[:, None] * sp
            + kv[None, :] * skv + head * sh + channel[None, :] * sd
        )
        destination_page = (destination_offset + j).to(tl.int64)
        destination_address = (
            destination + layer * dl + destination_page[:, None] * dp
            + kv[None, :] * dkv + head * dh + channel[None, :] * dd
        )
        if TWO_SOURCES:
            is_cold = source_page >= HOT_LENGTH
            cold_address = (
                cold_source + layer * cl + (source_page[:, None] - HOT_LENGTH) * cp
                + kv[None, :] * ckv + head * ch + channel[None, :] * cd
            )
            hot_values = tl.load(source_address, mask=mask & ~is_cold[:, None], other=0)
            cold_values = tl.load(cold_address, mask=mask & is_cold[:, None], other=0)
            values = tl.where(is_cold[:, None], cold_values, hot_values)
        else:
            values = tl.load(source_address, mask=mask, other=0)
        tl.store(destination_address, values, mask=mask)


@lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def pack_selected_kv(
    source, selected_pages, destination, count, *, destination_offset=0,
    num_programs=None, num_warps=8, tokens_per_tile=8,
    cold_source=None, source_hot_length=None,
):
    """Copy selected KV on the current stream of source.device; return None.

    source/destination: [layers, capacity, 2, 1, local_kv_heads, head_dim].
    selected_pages: [layers, local_kv_heads, selection_capacity], int32/int64.
    Read only the first count entries per layer/head; write consecutive pages
    starting at destination_offset. Existing tensor strides are respected.

    A destination may also be a pinned CPU tensor. CUDA's mapped host pointer
    lets this kernel gather directly into contiguous CPU backing, without a
    GPU staging allocation or a subsequent D2H copy. The caller must wait for
    a completion event before reading/reusing those CPU bytes. An unpinned
    CPU tensor is rejected. num_programs=None uses one program per device SM;
    an explicit program budget is retained for overlap experiments.

    No allocation, host/device synchronization, collectives, or page ownership
    changes occur here. TP ranks pass their own local heads and physical page
    numbers. This kernel does not determine when every rank may next forward.

    Caller obligations: selected entries must be valid source page numbers;
    source/selection writes must precede this launch; source and destination
    must stay alive and unmodified until this operation finishes. When using
    separate streams, establish producer/consumer dependencies with events.

    Counts and capacity-dependent strides are runtime kernel arguments, so a
    new history length does not compile a new kernel. Graph launch parameters
    and storage pointers must still remain valid throughout graph replay.

    With cold_source, indices are temporary slots in [hot[:source_hot_length],
    cold], independently per layer/head. Hot/cold may occupy disjoint intervals
    of the same arena as destination. Overlapping byte ranges are forbidden.
    """
    if source.ndim != 6 or destination.ndim != 6:
        raise ValueError("KV tensors must have six dimensions")
    layers, capacity, kv, page_size, heads, dim = source.shape
    if kv != 2 or page_size != 1 or min(layers, capacity, heads, dim) <= 0:
        raise ValueError("Expected nonempty KV tensors with K/V=2 and page_size=1")
    if destination.shape[0] != layers or destination.shape[2:] != source.shape[2:]:
        raise ValueError("Source and destination KV geometry must match")
    if selected_pages.ndim != 3 or selected_pages.shape[:2] != (layers, heads):
        raise ValueError("Selection shape must be [layers, local_heads, capacity]")
    if type(count) is not int or not 0 <= count <= selected_pages.shape[2]:
        raise ValueError("Invalid effective selection count")
    if (type(destination_offset) is not int or destination_offset < 0
            or destination_offset + count > destination.shape[1]):
        raise ValueError("Destination region is too small")
    host_destination = destination.device.type == "cpu" and (
        destination.is_pinned() or (count == 0 and destination.numel() == 0))
    if (source.device.type != "cuda" or selected_pages.device != source.device
            or not (destination.device == source.device or host_destination)):
        raise ValueError("Use CUDA sources/indices and a local CUDA or pinned CPU destination")
    if (source.dtype != destination.dtype
            or source.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError("Source and destination must use the same floating dtype")
    if selected_pages.dtype not in (torch.int32, torch.int64):
        raise ValueError("Physical page indices must be int32 or int64")
    if not source.is_contiguous() or not destination.is_contiguous():
        raise ValueError("KV regions must be contiguous in the original layout")
    def overlaps(first, second):
        return (first.nbytes and second.nbytes
                and first.data_ptr() < second.data_ptr() + second.nbytes
                and second.data_ptr() < first.data_ptr() + first.nbytes)

    if overlaps(source, destination):
        raise ValueError("Packing source and destination must not overlap")
    two_sources = cold_source is not None
    if two_sources:
        if (cold_source.ndim != 6 or cold_source.shape[0] != layers
                or cold_source.shape[2:] != source.shape[2:]
                or cold_source.device != source.device or cold_source.dtype != source.dtype
                or not cold_source.is_contiguous()):
            raise ValueError("Cold source must match rank-local hot KV geometry/device/dtype")
        if type(source_hot_length) is not int or not 0 < source_hot_length <= capacity:
            raise ValueError("Invalid effective source hot length")
        if overlaps(cold_source, destination):
            raise ValueError("Packing cold source and destination must not overlap")
    elif source_hot_length is not None:
        raise ValueError("source_hot_length requires a cold source")
    else:
        cold_source = source  # Unused pointer/strides, removed by constexpr dispatch.
        source_hot_length = 0
    if num_programs is not None and (type(num_programs) is not int or num_programs < 1):
        raise ValueError("num_programs must be a positive integer")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    if (type(tokens_per_tile) is not int or tokens_per_tile < 1
            or tokens_per_tile & (tokens_per_tile - 1)):
        raise ValueError("tokens_per_tile must be a positive power of two")
    if count == 0:
        return

    total_tasks = layers * heads * triton.cdiv(count, tokens_per_tile)
    programs = _sm_count(source.device.index) if num_programs is None else num_programs
    with torch.cuda.device(source.device):
        _pack_selected_kv[(min(programs, total_tasks),)](
            source, cold_source, selected_pages, destination, count, destination_offset,
            layers, heads, dim,
            source.stride(0), source.stride(1), source.stride(2),
            source.stride(4), source.stride(5),
            *selected_pages.stride(),
            destination.stride(0), destination.stride(1), destination.stride(2),
            destination.stride(4), destination.stride(5),
            cold_source.stride(0), cold_source.stride(1), cold_source.stride(2),
            cold_source.stride(4), cold_source.stride(5),
            TWO_SOURCES=two_sources, HOT_LENGTH=source_hot_length,
            TOKENS=tokens_per_tile, WIDTH=triton.next_power_of_2(2 * dim),
            num_warps=num_warps,
        )


@triton.jit
def _scatter_kv_pages(
    source, selection, destination, destination_pages,
    count, heads: tl.constexpr, dim: tl.constexpr,
    sl, sp: tl.constexpr, skv: tl.constexpr, sh: tl.constexpr,
    il, ih, dl, dp: tl.constexpr, dkv: tl.constexpr, dh: tl.constexpr,
    RESTORE: tl.constexpr, TOKENS: tl.constexpr, WIDTH: tl.constexpr,
):
    layer = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)
    j = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    lane = tl.arange(0, WIDTH)
    if RESTORE:
        src_page = j.to(tl.int64)
    else:
        src_page = tl.load(selection + layer * il + head * ih + j,
                           mask=j < count, other=0).to(tl.int64)
    dst_page = tl.load(destination_pages + j, mask=j < count, other=0).to(tl.int64)
    kv, channel = lane // dim, lane % dim
    mask = (j[:, None] < count) & (lane[None, :] < 2 * dim)
    value = tl.load(source + layer * sl + src_page[:, None] * sp
                    + kv[None, :] * skv + head * sh + channel[None, :], mask=mask, other=0)
    tl.store(destination + layer * dl + dst_page[:, None] * dp
             + kv[None, :] * dkv + head * dh + channel[None, :], value, mask=mask)


def _pool_copy(source, destination, destination_page_ids, selection=None):
    """Internal: callers validate ownership before submitting this stream work."""
    if (source.ndim != 6 or destination.ndim != 6 or source.shape[2:4] != (2, 1)
            or source.shape[0] != destination.shape[0] or source.shape[2:] != destination.shape[2:]
            or not source.is_contiguous() or not destination.is_contiguous()
            or source.dtype != destination.dtype
            or source.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or destination.device.type != "cuda"):
        raise ValueError("Expected matching contiguous original-layout KV and a CUDA pool")
    pages = tuple(destination_page_ids)
    if (len(set(pages)) != len(pages)
            or any(type(p) is not int or not 0 <= p < destination.shape[1] for p in pages)):
        raise ValueError("Destination pages must be unique valid pool pages")
    restore = selection is None
    if restore:
        if source.device.type != "cpu" or (source.numel() and not source.is_pinned()):
            raise ValueError("Restore requires mapped pinned CPU backing")
        if source.shape[1] != len(pages):
            raise ValueError("Restore must cover all cold slots")
        selection = destination  # Unused pointer; erased by constexpr dispatch.
        il = ih = 0
    else:
        if (source.device != destination.device or selection.device != source.device
                or selection.dtype not in (torch.int32, torch.int64) or selection.ndim != 3
                or selection.shape[:2] != (source.shape[0], source.shape[4])
                or selection.shape[-1] < len(pages) or selection.stride(-1) != 1):
            raise ValueError("Selection must be CUDA [layers, local heads, count]")
        il, ih = selection.stride()[:2]
    if not pages:
        return
    indices = torch.tensor(pages, device=destination.device, dtype=torch.int32)
    with torch.cuda.device(destination.device):
        _scatter_kv_pages[(triton.cdiv(len(pages), 8), source.shape[0], source.shape[4])](
            source, selection, destination, indices, len(pages), source.shape[4], source.shape[5],
            source.stride(0), source.stride(1), source.stride(2), source.stride(4), il, ih,
            destination.stride(0), destination.stride(1), destination.stride(2), destination.stride(4),
            RESTORE=restore, TOKENS=8, WIDTH=triton.next_power_of_2(2 * source.shape[5]),
            num_warps=8,
        )


def gather_to_pool(pool, selected_pages, destination_page_ids, *, source_page_ids):
    """Gather each layer/head into disjoint pages of the SAME original GPU pool.

    The CPU ownership lists must describe the frozen selection source and the
    newly allocated destination. Selected entries must belong to that source;
    the native TopK plus recent suffix establishes this without a CPU readback.
    Debug callers should separately validate selection membership/uniqueness.
    No pages are freed here. Caller retains tensors and waits for completion.
    """
    sources, destinations = set(source_page_ids), tuple(destination_page_ids)
    if any(type(p) is not int or not 0 <= p < pool.shape[1] for p in sources):
        raise ValueError("Invalid source ownership")
    if sources.intersection(destinations):
        raise ValueError("Same-pool gather cannot overwrite any live source page")
    _pool_copy(pool, pool, destinations, selected_pages)


def restore_to_pool(cold_cpu, pool, destination_page_ids):
    """Scatter mapped pinned CPU slots directly into original GPU pool pages.

    Caller allocates destination pages, keeps cold_cpu alive, and waits for an
    event before publishing the combined verify page table. No GPU staging.
    """
    _pool_copy(cold_cpu, pool, destination_page_ids)
