"""Small, aligned Top-K records of actual proposal/rejection randomness.

Native full-vocabulary sampling runs unchanged. E is kept in GPU memory only
until its verify; disk stores Q-topK, P-topK and the sampled token's union.
"""
from collections import deque
import importlib
import json
from pathlib import Path

import numpy as np
import torch


def round_limit_class(base, limit):
    class RoundLimitedScheduler(base):
        def is_request_complete(self, request, check_idx=-1):
            if super().is_request_complete(request, check_idx):
                return True
            verified_status = "STALL" if self.async_on else "VERIFY"
            if (request.status.name == verified_status
                    and self.acceptance_rounds.get(request.request_id, 0) >= limit):
                request.resource_finish_reason = "verify_round_limit"
                return True
            return False
    return RoundLimitedScheduler


def extend(base, top_k=100):
    class TopKTraceObserver(base):
        def __init__(self, scheduler, output, rank):
            super().__init__(scheduler, output, rank)
            self.top_trace_enabled = rank == 0
            if not self.top_trace_enabled:
                return
            self.top_k = min(top_k, scheduler.kv_pool.vocab_size)
            self.noise, self.proposal_context, self.top_pending = {}, [], deque()
            self.top_count = self.top_positions = self.top_waits = 0
            self.top_path = Path(output) / "topk-rounds"
            self.top_path.mkdir()
            self.top_meta = (Path(output) / "topk-trace.rank0.jsonl").open("x")
            self.original_prepare = scheduler.prepare_forward_inputs
            self.sampler_module = importlib.import_module(type(scheduler.sampler).__module__)
            self.original_fused = self.sampler_module.triton_fused_sampling
            scheduler.prepare_forward_inputs = self.prepare
            self.sampler_module.triton_fused_sampling = self.fused_sampling

        def prepare(self):
            host, tensors = self.original_prepare()
            self.proposal_context = []
            if host is not None:
                sampled_rows = getattr(self.owner, "_sample_rows", range(len(host["token_ids"])))
                mapping = {int(row): i for i, row in enumerate(sampled_rows)}
                for i, request in enumerate(self.owner.active_requests):
                    if request.status.name != "DRAFT":
                        continue
                    start, end = host["token_indptr"][i:i + 2]
                    if end - start != 1:
                        raise RuntimeError("Top-K trace requires one query per draft step")
                    self.proposal_context.append(dict(
                        request_id=request.request_id,
                        round=self.owner.acceptance_rounds.get(request.request_id, 0),
                        position=request.len_draft_probs - 1, sampler_row=mapping[int(start)],
                        forward_step=self.owner.step_id))
            return host, tensors

        def fused_sampling(self, logits, q, temperature, tp_ranks=1):
            ids, probs = self.original_fused(logits, q, temperature, tp_ranks)
            if self.proposal_context:
                rows = torch.tensor([row["sampler_row"] for row in self.proposal_context],
                                    dtype=torch.long, device=q.device)
                # The native sampler calls its Exp(1) tensor q; it is E here.
                # Copy only DRAFT rows, avoiding retention of large padded or
                # verify sampling tensors. No new random values are generated.
                noise = q.index_select(0, rows)
                token_ids = ids.index_select(0, rows)
                for i, row in enumerate(self.proposal_context):
                    key = row["request_id"], row["round"], row["position"]
                    if key in self.noise:
                        raise RuntimeError("Duplicate draft position in noise trace")
                    self.noise[key] = (noise[i], token_ids[i], dict(row))
            return ids, probs

        def sampling(self, **kwargs):
            result = super().sampling(**kwargs)
            if not self.top_trace_enabled:
                return result
            context = [dict(row) for row in self.prob_work[-1][0]]
            q, p, ids = kwargs["draft_probs"], kwargs["target_probs"], kwargs["draft_token_ids"]
            u = self.uniform_work[-1][1]
            q_values, q_ids = q.topk(self.top_k, dim=-1, sorted=True)
            p_values, p_ids = p.topk(self.top_k, dim=-1, sorted=True)
            candidates = torch.cat((q_ids, p_ids, ids.long().unsqueeze(-1)), dim=-1)
            q_candidates, p_candidates = q.gather(1, candidates), p.gather(1, candidates)
            e_candidates, actual_ids, steps = [], [], []
            row_index = 0
            control = torch.cuda.current_stream()
            for row in context:
                for j in range(row["actual_draft_tokens"]):
                    e, token, source = self.noise.pop((row["request_id"], row["round"], j))
                    # E was allocated on the compute stream and consumed here
                    # on the control stream. Keep its storage alive through use.
                    e.record_stream(control)
                    token.record_stream(control)
                    e_candidates.append(e.index_select(0, candidates[row_index]))
                    actual_ids.append(token)
                    steps.append(source["forward_step"])
                    row_index += 1
            tensors = dict(candidate_ids=candidates.to(torch.int32), q=q_candidates, p=p_candidates,
                           e=torch.stack(e_candidates), q_top_ids=q_ids.to(torch.int32),
                           p_top_ids=p_ids.to(torch.int32), q_top_mass=q_values.sum(-1),
                           p_top_mass=p_values.sum(-1), sampled_ids=ids,
                           draft_sampled_ids=torch.stack(actual_ids), u=u)
            if len(self.top_pending) >= 8:
                self.top_waits += 1
                self.top_pending[0][0].synchronize()
                self.drain_top()
            host = {}
            for name, tensor in tensors.items():
                cpu = torch.empty(tuple(tensor.shape), dtype=tensor.dtype,
                                  device="cpu", pin_memory=True)
                cpu.copy_(tensor, non_blocking=True)
                host[name] = cpu
            self.top_pending.append((control.record_event(), context, host, tensors, steps))
            self.drain_top()
            return result

        def drain_top(self, force=False):
            while self.top_pending:
                ready, context, host, keepalive, steps = self.top_pending[0]
                if force:
                    ready.synchronize()
                elif not ready.query():
                    break
                self.top_pending.popleft()
                arrays = {name: tensor.numpy() for name, tensor in host.items()}
                if not np.array_equal(arrays["sampled_ids"], arrays["draft_sampled_ids"]):
                    raise RuntimeError("Draft proposal IDs do not match verify IDs")
                offset = 0
                for row in context:
                    n = row["actual_draft_tokens"]
                    trace = {name: values[offset:offset + n].copy() for name, values in arrays.items()}
                    trace.pop("draft_sampled_ids")
                    # Sort and deduplicate the two top-K lists plus the sampled
                    # token. Padding is explicit, never renormalize Q or P.
                    width = 2 * self.top_k + 1
                    union_ids = np.full((n, width), -1, dtype=np.int32)
                    values = {name: np.full((n, width), np.nan, dtype=np.float32)
                              for name in ("q", "p", "e")}
                    valid = np.empty(n, dtype=np.int32)
                    for j in range(n):
                        unique, first = np.unique(trace["candidate_ids"][j], return_index=True)
                        valid[j] = len(unique)
                        union_ids[j, :len(unique)] = unique
                        for name in values:
                            values[name][j, :len(unique)] = trace[name][j, first]
                    trace.update(candidate_ids=union_ids, candidate_count=valid, **values,
                                 draft_forward_steps=np.asarray(steps[offset:offset + n], dtype=np.int64))
                    name = f"{self.top_count:06d}.npz"
                    with (self.top_path / name).open("xb") as file:
                        np.savez(file, **trace)
                    record = dict(row, archive=f"topk-rounds/{name}", top_k=self.top_k,
                                  positions=n, available="Q/E after draft; P/U after verify")
                    self.top_meta.write(json.dumps(record, separators=(",", ":")) + "\n")
                    self.top_count += 1
                    self.top_positions += n
                    offset += n

        def submit(self):
            super().submit()
            if self.top_trace_enabled:
                self.drain_top()

        def flush(self):
            super().flush()
            if self.top_trace_enabled:
                self.drain_top()
                self.top_meta.flush()

        def close(self, success):
            if self.closed:
                return
            finished = False
            try:
                super().close(success)
                if self.top_trace_enabled:
                    self.drain_top(force=True)
                finished = True
            finally:
                if self.top_trace_enabled:
                    self.owner.prepare_forward_inputs = self.original_prepare
                    self.sampler_module.triton_fused_sampling = self.original_fused
                    self.top_meta.close()
                    report = dict(successful=bool(success and finished), rank=0, top_k=self.top_k,
                                  rounds=self.top_count, positions=self.top_positions,
                                  terminal_unverified_noise_rows=len(self.noise),
                                  readback_queue_waits=self.top_waits,
                                  candidate_set="union(Q top-K, P top-K, actually sampled token)",
                                  values="actual float32 E/Q/P; probabilities not renormalized; U float64",
                                  noise="actual native Exp(1), retained until verify then released",
                                  no_extra_rng_draws=True,
                                  warning="Top-K excludes remaining vocabulary; diagnostic timing is not serving throughput")
                    (self.output / "topk-trace-summary.rank0.json").write_text(json.dumps(report, indent=2) + "\n")
                    self.noise.clear()
    return TopKTraceObserver


def validate(output):
    output = Path(output)
    summary = json.loads((output / "topk-trace-summary.rank0.json").read_text())
    if not summary["successful"]:
        raise ValueError("Top-K observer did not finish successfully")
    metadata = [json.loads(line) for line in (output / "topk-trace.rank0.jsonl").read_text().splitlines()]
    rounds = {(r["request_id"], r["round"]): r for r in (
        json.loads(line) for line in (output / "rounds.jsonl").read_text().splitlines())}
    seen, positions, boundary = set(), 0, []
    for record in metadata:
        identity = record["request_id"], record["round"]
        if identity in seen or identity not in rounds:
            raise ValueError("Duplicate or missing Top-K round identity")
        seen.add(identity)
        with np.load(output / record["archive"], allow_pickle=False) as trace:
            n = rounds[identity]["actual_draft_tokens"]
            if trace["candidate_ids"].shape != (n, 2 * summary["top_k"] + 1):
                raise ValueError("Incorrect Top-K candidate shape")
            q_token, p_token = [], []
            for j in range(n):
                count = int(trace["candidate_count"][j])
                ids = trace["candidate_ids"][j, :count]
                if len(set(ids.tolist())) != count or np.any(ids < 0):
                    raise ValueError("Invalid Top-K union IDs")
                if not set(trace["q_top_ids"][j]).union(trace["p_top_ids"][j]).issubset(ids):
                    raise ValueError("Top-K union is incomplete")
                for name in ("q", "p", "e"):
                    values = trace[name][j, :count]
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError(f"Invalid recorded {name}")
                e, q = trace["e"][j, :count], trace["q"][j, :count]
                if np.any(e <= 0):
                    raise ValueError("Invalid actual exponential noise")
                selected = np.flatnonzero(ids == trace["sampled_ids"][j])
                if len(selected) != 1:
                    raise ValueError("Actual sampled token missing from union")
                slot = int(selected[0])
                scores = q / e
                if scores[slot] != scores.max():
                    raise ValueError("Recorded actual E/Q do not reproduce the sampled winner within the union")
                q_token.append(q[slot])
                p_token.append(trace["p"][j, slot])
                for name, mass_name, ids_name in (("q", "q_top_mass", "q_top_ids"),
                                                 ("p", "p_top_mass", "p_top_ids")):
                    slots = np.searchsorted(ids, trace[ids_name][j])
                    top = trace[name][j, slots]
                    if np.any(np.diff(top) > 0) or not np.isclose(top.sum(), trace[mass_name][j], atol=2e-6):
                        raise ValueError("Incorrect Top-K order or probability mass")
            q = np.asarray(q_token, dtype=np.float32)
            p = np.asarray(p_token, dtype=np.float32)
            u = trace["u"]
            if u.dtype != np.float64 or not np.isfinite(u).all() or np.any((u < 0) | (u >= 1)):
                raise ValueError("Invalid actual rejection U")
            ratio = np.divide(p, q, out=np.zeros_like(p), where=q > 0)
            failed = np.flatnonzero((q <= 0) | (u > ratio.astype(np.float64)))
            accepted = int(failed[0]) if len(failed) else n
            expected = rounds[identity]["accepted_draft_tokens"]
            if accepted != expected:
                pos = min(accepted, expected)
                if abs(float(ratio[pos]) - float(u[pos])) > 1e-6:
                    raise ValueError("Top-K sampled P/Q/U do not reproduce native acceptance")
                boundary.append(dict(request_id=identity[0], round=identity[1], position=pos))
            positions += n
    if seen != set(rounds) or positions != summary["positions"] or len(seen) != summary["rounds"]:
        raise ValueError("Missing Top-K rounds or positions")
    payload = json.loads((output / "worker-payload.json").read_text())
    limit = payload.get("max_verify_rounds")
    audit = json.loads((output / "request-audit.json").read_text())
    per_request = []
    for row in audit:
        count = sum(key[0] == row["request_id"] for key in seen)
        if limit is not None and (count > limit or (
                row["finish_reason"] == "verify_round_limit" and count != limit)):
            raise ValueError("Per-request verify round cap was not respected")
        per_request.append(dict(request_id=row["request_id"], sample_id=row["sample_id"],
                                rounds=count, finish_reason=row["finish_reason"]))
    result = dict(successful=True, rounds=len(seen), positions=positions, top_k=summary["top_k"],
                  per_request=per_request, near_comparison_boundary=boundary,
                  checks="all recorded positions: noise winner within union, Top-K membership/mass, rejection and round cap")
    (output / "topk-trace-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))
