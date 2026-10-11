"""Summarize completed, identical-workload reproduction cases without mixing timings."""
import argparse
import csv
import json
from pathlib import Path
import re

DATASETS = ['aime24', 'olympiadbench', 'livecodebench']
REPO = Path(__file__).resolve().parents[2]


def read_case(case):
    launch = json.loads((case / 'launch.json').read_text())
    if launch['state'] != 'complete':
        raise ValueError(f'Case is not complete: {case}')
    workload = json.loads((case / 'engine/workload.json').read_text())
    first_requests = json.loads((case / 'engine/requests.json').read_text())
    if launch['engine'] == 'sparsespec':
        index = case / 'engine/epoch_index.json'
        epochs = json.loads(index.read_text()) if index.exists() else [dict(result_dir=str(case / 'engine'))]
        seconds, requests, accepted, proposed = 0.0, [], 0.0, 0.0
        for epoch in epochs:
            directory = Path(epoch['result_dir'])
            raw = (directory / 'summary.log').read_text()
            metrics = {label.strip(): float(number) for label, number in
                       re.findall(r'\|\s*([^|\n]+?)\s*\|\s*([\d.eE+-]+)\s*\|', raw)}
            seconds += metrics['time (s)']
            requests.extend(json.loads((directory / 'requests.json').read_text()))
            if epoch.get('counters_cumulative', True):
                accepted = metrics['Accepted Speculative Tokens']
                proposed = metrics['Total Speculative Tokens']
            else:
                accepted += metrics['Accepted Speculative Tokens']
                proposed += metrics['Total Speculative Tokens']
        # The original logger accumulates speculation counters across calls;
        # the final epoch's counters cover the whole case.
        acceptance = accepted / proposed if proposed else None
        first_requests = json.loads((case / 'engine/requests.json').read_text())
    else:
        metrics = json.loads((case / 'engine/summary.json').read_text())
        seconds, acceptance = metrics['engine_seconds'], None
        requests = first_requests
        first_requests = [request for request in requests if request['epoch'] == 0]
        epochs = sorted({request['epoch'] for request in requests})
    if len(first_requests) != workload['num_requests']:
        raise ValueError(f'Request count mismatch: {case}')
    initial_input_tokens = sum(request['len_input'] for request in first_requests)
    input_tokens = sum(request['len_input'] for request in requests)
    output_tokens = sum(request['len_output'] for request in requests)
    if initial_input_tokens != workload['input_tokens']:
        raise ValueError(f'Input token count mismatch: {case}')
    quality_paths = list((case / 'evaluation').rglob('*_metrics.json'))
    quality = json.loads(quality_paths[0].read_text()).get('acc') if len(quality_paths) == 1 else None
    # Official evaluator sets a dummy zero for LCB; it does not run code tests.
    if launch['dataset'] == 'livecodebench':
        quality = None
    return dict(engine=launch['engine'], dataset=launch['dataset'], requests=workload['num_requests'],
                generation_requests=len(requests), generation_epochs=len(epochs),
                max_batch_size=launch['max_batch_size'], mem_fraction=launch['mem_fraction'],
                dtype=launch['dtype'], temperature=launch['temperature'],
                engine_seconds=seconds, initial_input_tokens=initial_input_tokens,
                input_tokens=input_tokens, output_tokens=output_tokens,
                avg_input_tokens=initial_input_tokens / workload['num_requests'],
                avg_output_tokens=output_tokens / workload['num_requests'],
                output_tps=output_tokens / seconds, output_tps_per_gpu=output_tokens / seconds / 2,
                acceptance_rate=acceptance, accuracy=quality, case_path=str(case),
                resumed_generation_source=launch.get('resumed_generation_source'),
                prompt_sha256=workload['prompt_sha256'], token_ids_sha256=workload['token_ids_sha256'])


def summarize(base, phase='full'):
    base = Path(base).resolve()
    if not base.is_relative_to(REPO / 'log'):
        raise ValueError('Keep all reproduction results under SparseSpec/log')
    results, comparisons, pending = [], [], []
    for data in DATASETS:
        pair = {}
        for engine in ['sparsespec', 'vllm']:
            case = base / f'{engine}-{data}-{phase}'
            if not (case / 'launch.json').exists():
                pending.append(str(case))
                continue
            if json.loads((case / 'launch.json').read_text())['state'] != 'complete':
                pending.append(str(case))
                continue
            pair[engine] = read_case(case)
            results.append(pair[engine])
        if len(pair) != 2:
            continue
        s, v = pair['sparsespec'], pair['vllm']
        for field in ['requests', 'max_batch_size', 'mem_fraction', 'dtype', 'temperature',
                      'prompt_sha256', 'token_ids_sha256', 'initial_input_tokens']:
            if s[field] != v[field]:
                raise ValueError(f'{data}: unequal {field}; do not compute a speedup')
        comparisons.append(dict(dataset=data, output_throughput_ratio=s['output_tps'] / v['output_tps'],
                                sparsespec_output_tps=s['output_tps'], vllm_output_tps=v['output_tps'],
                                sparsespec_mean_output=s['avg_output_tokens'], vllm_mean_output=v['avg_output_tokens']))
    report = dict(phase=phase, complete=len(results) == 6, cases=results, comparisons=comparisons, pending=pending,
                  timing_note='SparseSpec engine summary excludes initialization; vLLM LLM.generate includes tokenization. Both include prefill and decode; quality evaluation is excluded.',
                  hardware_note='A100 PCIe 40GB TP2, batch cap128, official script temperature0.6; not the H100 paper setup.')
    (base / f'{phase}-comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if results:
        with (base / f'{phase}-cases.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    lines = ['# Qwen3-8B 对齐复现结果', '', f'阶段：{phase}；已完成 {len(results)}/6 组。', '',
             '| 数据集 | SparseSpec 输出 tok/s | vLLM 输出 tok/s | 吞吐比 | 平均输出长度 SparseSpec / vLLM |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in comparisons:
        lines.append(f'| {row["dataset"]} | {row["sparsespec_output_tps"]:.2f} | {row["vllm_output_tps"]:.2f} | '
                     f'{row["output_throughput_ratio"]:.3f} | {row["sparsespec_mean_output"]:.1f} / {row["vllm_mean_output"]:.1f} |')
    lines += ['', '两边 batch 上限128、TP2、FP16、温度0.6、thinking；使用相同请求与内存比例。',
              '统计包含 prefill 和 decode，排除模型加载、图捕获和质量评测。vLLM generate 还包含其内部 tokenization。',
              '吞吐比基于各自实际生成 token 数；同时查看输出长度和准确率，不把随机采样输出视为逐 token 一致。',
              '若评测流程追加生成轮次，累计各轮推理时间与 token 数；初始请求保持一致，后续工具反馈可能随采样结果变化。',
              '官方 LiveCodeBench 入口未运行代码测试，其占位 acc=0 不能作为准确率；本报告该指标记为空。',
              '硬件为 A100 PCIe 40GB；论文使用 H100、温度0.65、batch256。预检的输出上限1024，不用于正式结论。']
    (base / f'{phase}-comparison_zh.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['preflight', 'full'], default='full')
    args = parser.parse_args()
    report = summarize(args.root, args.phase)
    print(json.dumps(dict(complete=report['complete'], cases=len(report['cases']), comparisons=report['comparisons'])))
