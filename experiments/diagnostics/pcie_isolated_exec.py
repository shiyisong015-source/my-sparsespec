"""Run a TP1 profile with an idle PCIe-root lease and local CPU/memory binding."""
import argparse
import ctypes
import ctypes.util
import fcntl
import json
import os
from pathlib import Path
import sys


def pci_device(bus_id):
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode()
    domain, bus, slot = bus_id.lower().split(":")
    return Path('/sys/bus/pci/devices') / f'{int(domain,16):04x}:{bus}:{slot}'


def root_of(path):
    return next(parent.name for parent in path.resolve().parents if parent.name.startswith('pci0000:'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--lock-dir', type=Path, default=Path(__file__).resolve().parents[2]/'log/runtime/pcie-locks')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A worker command is required after --')
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(args.gpu)
    device = pci_device(nv.nvmlDeviceGetPciInfo(handle).busId)
    root = root_of(device)
    args.lock_dir.mkdir(parents=True, exist_ok=True)
    lease = (args.lock_dir/f'{root.replace(":","_")}.lock').open('a')
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'PCIe root {root} already leased by another profile')
    siblings = []
    for index in range(nv.nvmlDeviceGetCount()):
        other = nv.nvmlDeviceGetHandleByIndex(index)
        other_device = pci_device(nv.nvmlDeviceGetPciInfo(other).busId)
        if root_of(other_device) != root:
            continue
        pids = sorted({r.pid for r in (*nv.nvmlDeviceGetComputeRunningProcesses(other),
                                      *nv.nvmlDeviceGetGraphicsRunningProcesses(other))})
        siblings.append(dict(gpu=index, pci=other_device.name, pids=pids))
        if pids:
            raise SystemExit(f'PCIe root {root} occupied: GPU{index} PIDs={pids}; no worker started')
    nv.nvmlShutdown()
    node = int((device/'numa_node').read_text())
    if node < 0:
        raise SystemExit('GPU has no known NUMA node; refusing an unbound comparison')
    cpus = set()
    for group in (device/'local_cpulist').read_text().strip().split(','):
        bounds = list(map(int,group.split('-')))
        cpus.update(range(bounds[0],bounds[-1]+1))
    cpus &= os.sched_getaffinity(0)
    if not cpus:
        raise SystemExit('No allowed CPU local to GPU')
    os.sched_setaffinity(0,cpus)
    libnuma = ctypes.CDLL(ctypes.util.find_library('numa'), use_errno=True)
    libnuma.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    libnuma.set_mempolicy.restype = ctypes.c_int
    mask = ctypes.c_ulong(1 << node)
    if libnuma.set_mempolicy(2,ctypes.byref(mask),ctypes.sizeof(mask)*8) != 0:
        raise OSError(ctypes.get_errno(),'Cannot bind worker memory to the GPU-local NUMA node')
    placement = dict(gpu=args.gpu,pci=device.name,pcie_root=root,numa_node=node,
                     cpu_affinity=sorted(cpus),memory_policy='MPOL_BIND',siblings=siblings,
                     command=command, pid=os.getpid())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(placement,indent=2)+'\n')
    os.set_inheritable(lease.fileno(),True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
    print('PROFILE_PLACEMENT',json.dumps(placement),flush=True)
    os.execvpe(command[0],command,os.environ)


if __name__ == '__main__':
    main()
