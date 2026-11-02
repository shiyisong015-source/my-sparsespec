"""Actual proposal noise and full vocabulary probabilities for small diagnostics.

Only rank 0 writes full tensors. Native sampling/RNG calls are intercepted after
execution, never repeated. Q/E rows are recorded at DRAFT sampling; P and actual
U are recorded at verification. Separate row indexes join asynchronous stages.
"""
from collections import deque
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


class FullTraceWriter:
    def __init__(self, output, vocab_size, max_pending_bytes=256 << 20):
        self.output, self.vocab_size = Path(output), vocab_size
        self.pending, self.pending_bytes = deque(), 0
        self.max_pending_bytes = max_pending_bytes
        self.counts = dict(proposal=0, verify=0, verify_positions=0)
        self.waits = 0
        self.files = {name: (self.output / name).open("xb") for name in (
            "proposal-e.rank0.f32", "proposal-q.rank0.f32", "verify-p.rank0.f32")}
        self.metadata = {kind: (self.output / f"full-{kind}.rank0.jsonl").open("x")
                         for kind in ("proposal", "verify")}

    def enqueue(self, kind, context, tensors):
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        while self.pending and self.pending_bytes + size > self.max_pending_bytes:
            self.waits += 1
            self.pending[0][0].synchronize()
            self.drain()
        host = {}
        for name, tensor in tensors.items():
            cpu = torch.empty(tuple(tensor.shape), dtype=tensor.dtype,
                              device="cpu", pin_memory=True)
            cpu.copy_(tensor, non_blocking=True)
            host[name] = cpu
        ready = torch.cuda.current_stream().record_event()
        self.pending.append((ready, kind, context, host, tensors, size))
        self.pending_bytes += size
        self.drain()

    def drain(self, force=False):
        while self.pending:
            ready, kind, context, host, keepalive, size = self.pending[0]
            if force:
                ready.synchronize()
            elif not ready.query():
                break
            self.pending.popleft()
            self.pending_bytes -= size
            # Fail explicitly before exhausting the filesystem; never silently
            # truncate the requested diagnostic or change its generation cap.
            if shutil.disk_usage(self.output).free < size + (64 << 30):
                raise OSError("Full trace reached the 64-GiB free-disk reserve")
            arrays = {name: cpu.numpy() for name, cpu in host.items()}
            if kind == "proposal":
                start = self.counts["proposal"]
                arrays["e"].astype("<f4", copy=False).tofile(self.files["proposal-e.rank0.f32"])
                arrays["q"].astype("<f4", copy=False).tofile(self.files["proposal-q.rank0.f32"])
                for i, row in enumerate(context):
                    record = dict(row, binary_row=start + i, token_id=int(arrays["ids"][i]))
                    self.metadata[kind].write(json.dumps(record, separators=(",", ":")) + "\n")
                self.counts[kind] += len(context)
            else:
                arrays["p"].astype("<f4", copy=False).tofile(self.files["verify-p.rank0.f32"])
                offset = 0
                for row in context:
                    n = row["actual_draft_tokens"]
                    stop = offset + n
                    record = dict(row, binary_row=self.counts["verify_positions"],
                                  token_ids=arrays["ids"][offset:stop].tolist(),
                                  u=arrays["u"][offset:stop].tolist(),
                                  q_token=arrays["q_token"][offset:stop].tolist(),
                                  p_token=arrays["p_token"][offset:stop].tolist())
                    self.metadata[kind].write(json.dumps(record, separators=(",", ":")) + "\n")
                    self.counts["verify"] += 1
                    self.counts["verify_positions"] += n
                    offset = stop
                if offset != len(arrays["ids"]):
                    raise RuntimeError("Full verify trace has invalid request offsets")

    def flush(self):
        self.drain()
        for file in (*self.files.values(), *self.metadata.values()):
            file.flush()

    def close(self, success):
        finished = False
        try:
            self.drain(force=True)
            finished = True
        finally:
            for file in (*self.files.values(), *self.metadata.values()):
                file.close()
            report = dict(successful=bool(success and finished), vocab_size=self.vocab_size,
                          rank=0, counts=self.counts, readback_queue_waits=self.waits,
                          max_pending_bytes=self.max_pending_bytes,
                          matrix_dtype="little-endian float32", u_dtype="float64 in JSON and uniform file",
                          matrix_shape=["positions", self.vocab_size],
                          vocabulary_order="canonical token ID, not TP shard order",
                          proposal_key=["request_id", "round", "position"],
                          position_numbering="zero based within each draft window",
                          source="Actual native Exp(1) noise E and temperature-adjusted Q/P; no extra RNG draws",
                          terminal_drafts="Q/E kept even when EOS ends the request before another verify",
                          scope="Draft positions only; verify bonus and prefill outputs are not draft candidates",
                          timing="Heavy diagnostic readback/write overhead; not a throughput benchmark")
            (self.output / "full-trace-summary.rank0.json").write_text(json.dumps(report, indent=2) + "\n")


def extend(base):
    class FullVocabularyObserver(base):
        def __init__(self, scheduler, output, rank):
            super().__init__(scheduler, output, rank)
            self.full_writer = None
            if rank != 0:
                return
            self.full_writer = FullTraceWriter(output, scheduler.kv_pool.vocab_size)
            self.proposal_context = []
            self.sampler_module = importlib.import_module(type(scheduler.sampler).__module__)
            self.original_fused_sampling = self.sampler_module.triton_fused_sampling
            self.original_prepare = scheduler.prepare_forward_inputs
            scheduler.prepare_forward_inputs = self.prepare
            self.sampler_module.triton_fused_sampling = self.fused_sampling

        def prepare(self):
            host, tensors = self.original_prepare()
            self.proposal_context = []
            if host is not None:
                # Dynamic replay may filter sampler rows. Map the original
                # forward rows to the compact rows actually seen by Sampler.
                sampled_rows = getattr(self.owner, "_sample_rows", range(len(host["token_ids"])))
                sampled_index = {int(row): i for i, row in enumerate(sampled_rows)}
                for i, request in enumerate(self.owner.active_requests):
                    if request.status.name != "DRAFT":
                        continue
                    start, end = host["token_indptr"][i:i + 2]
                    if end - start != 1:
                        raise RuntimeError("A draft proposal must have exactly one query")
                    self.proposal_context.append(dict(
                        request_id=request.request_id,
                        round=self.owner.acceptance_rounds.get(request.request_id, 0),
                        position=request.len_draft_probs - 1,
                        forward_step=self.owner.step_id,
                        sampler_row=sampled_index[int(start)],
                        input_position=int(host["pos_offsets"][i]),
                        available="after_this_draft_forward"))
            return host, tensors

        def fused_sampling(self, logits, q, temperature, tp_ranks=1):
            # Native sampler calls its exponential noise tensor `q`; it is E,
            # while returned `probs` is the draft vocabulary distribution Q.
            ids, probs = self.original_fused_sampling(logits, q, temperature, tp_ranks)
            if self.proposal_context:
                context = [dict(row) for row in self.proposal_context]
                indices = torch.tensor([r["sampler_row"] for r in context],
                                       dtype=torch.long, device=q.device)
                self.full_writer.enqueue("proposal", context, dict(
                    e=q.index_select(0, indices), q=probs.index_select(0, indices),
                    ids=ids.index_select(0, indices)))
            return ids, probs

        def sampling(self, **kwargs):
            result = super().sampling(**kwargs)
            if self.full_writer is not None:
                context = [dict(row) for row in self.prob_work[-1][0]]
                uniform = self.uniform_work[-1][1]
                ids = kwargs["draft_token_ids"]
                positions = torch.arange(len(ids), device=ids.device)
                proposed = ids.long()
                self.full_writer.enqueue("verify", context, dict(
                    p=kwargs["target_probs"], ids=ids, u=uniform,
                    q_token=kwargs["draft_probs"][positions, proposed],
                    p_token=kwargs["target_probs"][positions, proposed]))
            return result

        def submit(self):
            super().submit()
            if self.full_writer is not None:
                self.full_writer.drain()

        def flush(self):
            super().flush()
            if self.full_writer is not None:
                self.full_writer.flush()

        def close(self, success):
            if self.closed:
                return
            finished = False
            try:
                super().close(success)
                finished = True
            finally:
                if self.full_writer is not None:
                    self.owner.prepare_forward_inputs = self.original_prepare
                    self.sampler_module.triton_fused_sampling = self.original_fused_sampling
                    self.full_writer.close(bool(success and finished))

    return FullVocabularyObserver


def read_trace(output):
    """Memory-map full matrices; returns metadata and arrays without loading GBs."""
    output = Path(output)
    summary = json.loads((output / "full-trace-summary.rank0.json").read_text())
    if not summary["successful"]:
        raise ValueError("The full trace did not finish successfully")
    metadata = {}
    for kind in ("proposal", "verify"):
        metadata[kind] = [json.loads(line) for line in (
            output / f"full-{kind}.rank0.jsonl").read_text().splitlines() if line.strip()]
    arrays = {}
    for name, filename, count in (
            ("e", "proposal-e.rank0.f32", summary["counts"]["proposal"]),
            ("q", "proposal-q.rank0.f32", summary["counts"]["proposal"]),
            ("p", "verify-p.rank0.f32", summary["counts"]["verify_positions"])):
        path = output / filename
        if path.stat().st_size != count * summary["vocab_size"] * 4:
            raise ValueError(f"Incorrect full matrix size: {path.name}")
        arrays[name] = np.memmap(path, dtype="<f4", mode="r",
                                 shape=(count, summary["vocab_size"]))
    return summary, metadata, arrays


def validate(output, full_rows=256):
    """Check all joins/scalars and a deterministic sample of whole vocab rows."""
    output = Path(output)
    summary, metadata, arrays = read_trace(output)
    proposals = metadata["proposal"]
    index = {}
    for i, row in enumerate(proposals):
        identity = row["request_id"], row["round"], row["position"]
        if row["binary_row"] != i or identity in index:
            raise ValueError("Duplicate or invalid full proposal position")
        index[identity] = row
    observed = {(r["request_id"], r["round"]): r for r in (
        json.loads(line) for line in (output / "rounds.jsonl").read_text().splitlines())}
    used, checked, target_rows = set(), [], 0
    for row in metadata["verify"]:
        identity = row["request_id"], row["round"]
        n = row["actual_draft_tokens"]
        if identity not in observed or n != observed[identity]["actual_draft_tokens"]:
            raise ValueError("Full verify/acceptance join failed")
        if row["binary_row"] != target_rows:
            raise ValueError("Non-contiguous full verify probability rows")
        target_rows += n
        q_rows = []
        for j, token in enumerate(row["token_ids"]):
            source = index[(*identity, j)]
            if source["token_id"] != token:
                raise ValueError("Draft noise/token identity differs from verify")
            q_rows.append(source["binary_row"])
            used.add(source["binary_row"])
        tokens = np.asarray(row["token_ids"], dtype=np.int64)
        if len(tokens) != n:
            raise ValueError("Full verify trace has incorrect token count")
        p_rows = np.arange(row["binary_row"], row["binary_row"] + n)
        q = arrays["q"][q_rows, tokens]
        p = arrays["p"][p_rows, tokens]
        if not (np.array_equal(q, np.asarray(row["q_token"], dtype=np.float32)) and
                np.array_equal(p, np.asarray(row["p_token"], dtype=np.float32))):
            raise ValueError("Full vocabulary probabilities differ from sampled-token probabilities")
        u = np.asarray(row["u"], dtype=np.float64)
        ratio = np.divide(p, q, out=np.zeros_like(p), where=q > 0)
        failed = np.flatnonzero((q <= 0) | (u > ratio.astype(np.float64)))
        accepted = int(failed[0]) if len(failed) else n
        if accepted != observed[identity]["accepted_draft_tokens"]:
            raise ValueError("Full trace does not reproduce native acceptance")
        checked.extend(zip(q_rows, p_rows))
    if len(metadata["verify"]) != len(observed) or target_rows != summary["counts"]["verify_positions"]:
        raise ValueError("Missing full verify rounds")
    sample = np.unique(np.linspace(0, len(checked) - 1, min(full_rows, len(checked)), dtype=int))
    for i in sample:
        q_row, p_row = checked[i]
        q, p, e = arrays["q"][q_row], arrays["p"][p_row], arrays["e"][q_row]
        if not (np.isfinite(e).all() and np.all(e > 0)):
            raise ValueError("Invalid actual exponential noise")
        for distribution in (q, p):
            if not (np.isfinite(distribution).all() and np.all(distribution >= 0)
                    and np.isclose(distribution.sum(dtype=np.float64), 1., atol=2e-5)):
                raise ValueError("Invalid full vocabulary probability distribution")
        scores = q / e
        if scores[proposals[q_row]["token_id"]] != scores.max():
            raise ValueError("Stored Q/E does not reproduce native proposal sampling")
    result = dict(successful=True, proposal_positions=len(proposals), verify_rounds=len(observed),
                  verified_positions=len(used), terminal_unverified_proposals=len(proposals) - len(used),
                  complete_vocab_rows_checked=len(sample), sampled_token_and_acceptance_checks="all verified positions",
                  vocab_size=summary["vocab_size"], dtype="float32, unchanged")
    (output / "full-trace-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-rows", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(validate(args.output, args.full_rows), indent=2))
