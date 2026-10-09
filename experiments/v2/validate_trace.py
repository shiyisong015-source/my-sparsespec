"""Check trace/output conservation and optional deterministic output equivalence."""
import argparse
import json
from pathlib import Path


def validate(directory):
    report=json.loads((directory/'case.json').read_text())
    output=json.loads((directory/'requests.json').read_text())
    assert len(output)==len(report['inputs'])
    assert sorted(r['id'] for r in output)==list(range(len(output)))
    if not report['disable_timing']:
        rows=[json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
        summary=json.loads((directory/'steps.summary.json').read_text())
        assert summary['successful']
        assert [r['step_id'] for r in rows]==list(range(len(rows)))
        assert sum(r['committed_tokens'] for r in rows)==sum(r['len_output'] for r in output)==summary['committed_tokens']
        for r in rows:
            assert r['active_batch']==len(r['requests'])
            assert sum(v['query_tokens'] for v in r['requests'])==r['query_tokens']
            assert r['step_wall_seconds']>=0 and all(v>=0 for v in r['gpu_ms'].values())
            assert sum(r['committed_by_request'].values())==r['committed_tokens']
        roundfile=directory/'rounds.jsonl'
        if roundfile.exists():
            for s in roundfile.read_text().splitlines():
                r=json.loads(s)
                step = rows[r['step_id']]
                key = str(r['request_id'])
                trimmed = step.get('eos_trimmed_by_request', {}).get(key, 0)
                assert step['committed_by_request'][key]==r['accepted_draft_tokens']+1-trimmed
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directories',type=Path,nargs='+')
    p.add_argument('--equal-outputs',action='store_true',help='Use only for an explicitly deterministic control')
    args=p.parse_args();outputs=[validate(d) for d in args.directories]
    if args.equal_outputs:
        expected=[r['output_token_ids'] for r in outputs[0]]
        assert all([r['output_token_ids'] for r in rows]==expected for rows in outputs[1:]), 'Outputs differ'
    print(json.dumps({'validated':list(map(str,args.directories)),'equal_outputs_checked':args.equal_outputs}))


if __name__=='__main__':main()
