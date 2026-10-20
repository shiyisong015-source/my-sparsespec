"""Full-query score correctness, native regression and graph replay checks."""
import math

import unittest
import torch

from serve.attention.paged_scores import prepare_paged_scores, paged_selection_scores

def make_case(dtype=torch.float16, kinds=(0, 1, 2, 2, 2), queries=(3, 1, 33, 17, 1),
              lengths=(101, 127, 8193, 257, 65)):
    torch.manual_seed(330)
    heads, group, dim = 4, 4, 128
    device = "cuda"
    qo = [0]
    kvp = [0]
    scores = []
    total = 0
    for kind, qn, length in zip(kinds, queries, lengths):
        qo.append(qo[-1] + qn)
        kvp.append(kvp[-1] + length)
        scores.append(total)
        total += length if kind == 2 else 0
    kv = torch.randn(sum(lengths)+13, 2, 1, heads, dim, device=device, dtype=dtype) * .5
    q = torch.randn(qo[-1], heads*group, dim, device=device, dtype=dtype) * .5
    pages = torch.randperm(kv.shape[0], device=device)[:sum(lengths)].int()
    lse = torch.empty(q.shape[:2], device=device, dtype=torch.float32)
    # Make the final query matter, detecting the original 16-query truncation.
    for i, kind in enumerate(kinds):
        if kind == 2:
            key = kv[pages[kvp[i]].long(), 0, 0]
            q[qo[i+1]-1] = key.repeat_interleave(group, dim=0) * 4
    metadata = dict(batch_size_host=len(kinds),
        batch_size_tensor=torch.tensor([len(kinds)], dtype=torch.int32, device=device),
        all_kv_indptr=torch.tensor(kvp, dtype=torch.int32, device=device),
        qo_indptr=torch.tensor(qo, dtype=torch.int32, device=device),
        kv_indices=pages,
        request_type=torch.tensor(kinds, dtype=torch.int32, device=device),
        verify_kv_indptr=torch.tensor(scores, dtype=torch.int32, device=device),
        score_tile_indptr=torch.empty(len(kinds)+1, dtype=torch.int32, device=device),
        dump_logits=torch.full((1, heads, total+64), -777., dtype=torch.float16, device=device))
    expected = torch.full_like(metadata["dump_logits"][0], -777.)
    for i, kind in enumerate(kinds):
        if kind != 2:
            continue
        qn, length = queries[i], lengths[i]
        query = q[qo[i]:qo[i+1]].float().reshape(qn, heads, group, dim)
        keys = kv[pages[kvp[i]:kvp[i+1]].long(), 0, 0].float()
        logits = torch.einsum("qhgd,nhd->qhgn", query, keys) / math.sqrt(dim)
        visible = (torch.arange(length, device=device)[None, :]
                   <= length-qn+torch.arange(qn, device=device)[:, None])
        logits = logits.masked_fill(~visible[:, None, None, :], -torch.inf)
        normalizer = torch.logsumexp(logits, dim=-1)
        lse[qo[i]:qo[i+1]] = normalizer.reshape(qn, heads*group) / math.log(2)
        probability = torch.softmax(logits, dim=-1)
        score = probability.sum(dim=(0, 2)) / (visible.sum(dim=0)[None, :] * group)
        expected[:, scores[i]:scores[i]+length] = score.half()
    return q, kv, lse, metadata, expected


def test_full_33_queries_match_independent_reference(dtype):
    q, kv, lse, m, expected = make_case(dtype)
    prepare_paged_scores(m)
    paged_selection_scores(q, kv, lse, m, 0, 33)
    torch.testing.assert_close(m["dump_logits"][0], expected, rtol=5e-3, atol=2e-6)


def test_native_small_window_score_regression():
    from serve.attention.backend import BatchTopKAttention
    q, kv, _, m, _ = make_case(kinds=(2, 2), queries=(9, 1), lengths=(1025, 65))
    wrapper = BatchTopKAttention(device="cuda", num_kv_heads=4, num_qo_heads=16,
        head_dim=128, q_data_type=torch.float16, kv_data_type=torch.float16,
        MAX_SPEC_STRIDE=9, MAX_TOTAL_DRAFT_KV_LEN=2048, MAX_TOTAL_VERIFY_KV_LEN=2048)
    lens = torch.diff(m["all_kv_indptr"])
    wrapper.plan(m["qo_indptr"], m["all_kv_indptr"], lens, 16, 4, 128, 128, 1,
                 causal=True, q_data_type=q.dtype, kv_data_type=kv.dtype)
    draft = torch.zeros((4, 2048), device="cuda", dtype=torch.int32)
    out, lse = wrapper.run_attention(q, kv, m["kv_indices"], draft, m["request_type"])
    legacy = torch.empty((4, 2048), device="cuda", dtype=torch.float16)
    wrapper.rematerialize_scores(q, out, lse, kv, m["kv_indices"], m["request_type"],
                                  m["verify_kv_indptr"], legacy)
    prepare_paged_scores(m)
    paged_selection_scores(q, kv, lse, m, 0, 33)
    torch.testing.assert_close(m["dump_logits"][0, :, :1090], legacy[:, :1090], rtol=5e-3, atol=2e-6)


def test_graph_replay_reads_current_batch_and_request_types():
    q, kv, lse, m, expected = make_case()
    prepare_paged_scores(m)
    for _ in range(3):
        paged_selection_scores(q, kv, lse, m, 0, 33)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_selection_scores(q, kv, lse, m, 0, 33)
    graph.replay()
    torch.testing.assert_close(m["dump_logits"][0], expected, rtol=5e-3, atol=2e-6)
    # First two requests are NORMAL/DRAFT: an empty score plan must be a no-op.
    m["batch_size_host"] = 2
    m["batch_size_tensor"].fill_(2)
    prepare_paged_scores(m)
    m["dump_logits"].fill_(-777.)
    graph.replay()
    assert (m["dump_logits"] == -777.).all().item()
    m["batch_size_host"] = 5
    m["batch_size_tensor"].fill_(5)
    prepare_paged_scores(m)
    graph.replay()
    torch.testing.assert_close(m["dump_logits"][0], expected, rtol=5e-3, atol=2e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PagedScoreTests(unittest.TestCase):
    def test_full_33_queries(self):
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                test_full_33_queries_match_independent_reference(dtype)

    def test_native_regression(self):
        test_native_small_window_score_regression()

    def test_long_context_boundary(self):
        q, kv, lse, m, expected = make_case(kinds=(2,), queries=(33,), lengths=(32769,))
        prepare_paged_scores(m)
        paged_selection_scores(q, kv, lse, m, 0, 33)
        torch.testing.assert_close(m["dump_logits"][0], expected, rtol=5e-3, atol=2e-6)

    def test_graph_replay(self):
        test_graph_replay_reads_current_batch_and_request_types()


if __name__ == "__main__":
    unittest.main()
