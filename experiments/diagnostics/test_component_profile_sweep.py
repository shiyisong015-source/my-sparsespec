"""CPU-only regressions for occupied GPUs and preserving failed scan attempts."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import pwd
import tempfile
import threading
import unittest
from unittest.mock import patch

import run_component_profile_sweep as sweep


def device(gpu, busy=False):
    return dict(gpu=gpu,cpu_affinity=[0],used_bytes=9*1024**3 if busy else 0,
                pids=[123] if busy else [])


class SweepTest(unittest.TestCase):
    def test_rotation_skips_occupied_gpu_and_waits_if_all_busy(self):
        self.assertEqual(sweep.select_idle_gpu([device(2,True),device(3)],[2,3],0),(3,2))
        self.assertEqual(sweep.select_idle_gpu([device(2,True),device(3,True)],[2,3],0),(None,0))
        self.assertEqual(sweep.select_idle_gpu([device(2),device(3)],[2,3],1),(3,2))

    def test_resume_skips_completed_and_rejects_new_batches(self):
        jobs={"16":{"status":"failed"},"32":{"status":"complete"},"48":{"status":"pending"}}
        self.assertEqual(sweep.resume_batches(jobs,[16,32,48]),[16,48])
        with self.assertRaises(ValueError):
            sweep.resume_batches(jobs,[64])

    def test_busy_guard_retry_preserves_previous_attempt_and_completed_case(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            root=base/"original"
            old_output=root/"jobs/b16"
            old_output.mkdir(parents=True)
            (old_output/"launcher.log").write_text("original failure\n")
            complete=dict(status="complete",gpu=6,output=str(root/"jobs/b32"),sampling_complete=True)
            plan=dict(record_windows=3,record_seconds=180.,window_gap_seconds=180.,buffer_batch_capacity=128,tp=1)
            (root/"sweep-status.json").write_text(json.dumps(dict(plan=plan,jobs={
                "16":dict(status="failed",gpu=2,output=str(old_output)),"32":complete})))
            args=argparse.Namespace(gpus=[2],batches=[16,32],dry_run=False,resume=root,
                wait_for_controller=False,windows=3,record_seconds=180.,window_gap_seconds=180.)
            calls=[]
            def popen(command,**kwargs):
                output=Path(kwargs["env"]["COMPONENT_PROFILE_OUTPUT"])
                calls.append(output)
                code=75 if len(calls)==1 else 0
                if code==0:
                    (output/"component-summary.json").write_text(json.dumps(dict(
                        sampling_coverage={"b16":{"status":"complete"}})))
                return argparse.Namespace(pid=1000+len(calls),wait=lambda:code)
            def run(command,**kwargs):
                if "mkdir" in command:
                    Path(command[-1]).mkdir(parents=True,exist_ok=True)
                return argparse.Namespace(returncode=0)
            original_wait=threading.Event.wait
            def fast_wait(event,timeout=None):
                return original_wait(event,0 if timeout==30 else timeout)
            identity=pwd.getpwuid(os.getuid())
            # Initial inventory occupied, first dispatch occupied, then free.
            inventories=iter([True,True,False,False])
            def placement(gpus):
                busy=next(inventories)
                return [device(2,busy)],[dict(gpus=[2],cpus=[0])]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sweep,"BASE",base))
                stack.enter_context(patch.object(sweep,"placement",side_effect=placement))
                stack.enter_context(patch.object(sweep.os,"geteuid",return_value=0))
                stack.enter_context(patch.object(sweep.os,"chown"))
                stack.enter_context(patch.dict(os.environ,SUDO_USER=identity.pw_name))
                stack.enter_context(patch.object(sweep.subprocess,"run",side_effect=run))
                stack.enter_context(patch.object(sweep.subprocess,"Popen",side_effect=popen))
                stack.enter_context(patch.object(threading.Event,"wait",fast_wait))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                sweep.main(args)
            result=json.loads((root/"sweep-status.json").read_text())
            self.assertEqual(result["jobs"]["32"],complete)
            self.assertEqual(result["jobs"]["16"]["status"],"complete")
            self.assertEqual(len(result["jobs"]["16"]["attempts"]),2)
            self.assertEqual(result["jobs"]["16"]["attempts"][-1]["returncode"],75)
            self.assertNotEqual(calls[0],calls[1])
            self.assertEqual((old_output/"launcher.log").read_text(),"original failure\n")


if __name__=="__main__":
    unittest.main()
