"""NVTX containment regression for nested ranges and unmatched CUDA APIs."""
import random
import json
from pathlib import Path
import tempfile
import unittest
from analyze_component_profile import EnclosingRangeIndex, host_transfer_intervals


class RangeTests(unittest.TestCase):
    def test_latest_start_enclosing_range(self):
        index=EnclosingRangeIndex([(0,100,'forward'),(10,90,'attention'),(20,30,'kernel')])
        self.assertEqual(index.enclosing(22,25),'kernel')
        self.assertEqual(index.enclosing(22,35),'attention')
        self.assertEqual(index.enclosing(95,99),'forward')
        self.assertIsNone(index.enclosing(101,102))
        self.assertIsNone(index.enclosing(-5,-1))

    def test_stable_ties_and_empty_ranges(self):
        index=EnclosingRangeIndex([(0,100,'first'),(0,50,'second')])
        self.assertEqual(index.enclosing(1,25),'second')
        self.assertEqual(index.enclosing(1,70),'first')
        self.assertIsNone(EnclosingRangeIndex([]).enclosing(0,1))

    def test_crossing_and_unmatched_ranges_match_containment_oracle(self):
        rng=random.Random(330)
        values=sorted([(start:=rng.randrange(10000),start+rng.randrange(1,500),i)
                       for i in range(1500)],key=lambda row:row[0])
        index=EnclosingRangeIndex(values)
        for _ in range(1000):
            start=rng.randrange(-100,11000)
            end=start+rng.randrange(1,600)
            expected=next((label for a,b,label in reversed(values) if a<=start and end<=b),None)
            self.assertEqual(index.enclosing(start,end),expected)


class TransferIntervalTests(unittest.TestCase):
    def test_chunks_match_repeated_offsets_within_one_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            case=root/'b32_w02';case.mkdir()
            (root/'b32').mkdir()
            rows=[]
            for window,start,end in ((1,10,20),(2,30,40),(2,50,60)):
                for event,time in (('chunk_submit',start),('chunk_complete',end)):
                    rows.append(dict(event=event,cpu_ns=time,window_id=window,
                        request_id=1,generation=2,direction='up',offset=0))
            # A completion with no observed submit must not invent an interval.
            rows.append(dict(event='chunk_complete',cpu_ns=70,window_id=2,
                request_id=2,generation=2,direction='down',offset=0))
            (root/'b32/distance-swap.rank0.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in rows))
            found=host_transfer_intervals(root,case,0)
            self.assertEqual([(r['submit_cpu_ns'],r['commit_cpu_ns'],r['direction'])
                             for r in found],[(30,40,'restore'),(50,60,'restore')])


if __name__=='__main__':
    unittest.main()
