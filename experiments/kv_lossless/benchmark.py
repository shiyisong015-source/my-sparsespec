"""Bitwise lossless codec benchmarks on captured, used cold KV (no model changes)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PACKAGES = REPO / ".repro/kv-codec-packages"
sys.path.insert(0, str(PACKAGES))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def summary(seconds, count):
    return dict(median_ms=float(np.median(seconds) * 1000),
                max_ms=float(max(seconds) * 1000), repeats=len(seconds),
                raw_GB_per_second=float(count / np.median(seconds) / 1e9))


def timed(fn, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def shuffle(raw):
    return np.frombuffer(raw, np.uint8).reshape(-1, 2).T.copy().tobytes()


def unshuffle(raw):
    return np.frombuffer(raw, np.uint8).reshape(2, -1).T.copy().tobytes()


def inputs(directory):
    for manifest in sorted(directory.glob("capture.rank*.json")):
        for row in json.loads(manifest.read_text()):
            array = np.load(directory / row["file"], allow_pickle=False)
            if hashlib.sha256(array.tobytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("Captured KV checksum mismatch")
            yield row, array


def cpu(directory, output):
    import lz4.block
    import zstandard as zstd
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row, array in inputs(directory):
        # K and V are evaluated independently; their sum is the stored KV cost.
        for kind, index in [("K", 0), ("V", 1)]:
            raw = np.ascontiguousarray(array[:, :, index]).tobytes()
            for block_size in [65536, 1048576]:
                blocks = [raw[i:i + block_size] for i in range(0, len(raw), block_size)]
                for transform in ["none", "byte_shuffle"]:
                    for name in ["LZ4", "Zstd1", "Zstd3"]:
                        if name == "LZ4":
                            encode = lambda x: lz4.block.compress(x, store_size=True)
                            decode = lz4.block.decompress
                        else:
                            compressor = zstd.ZstdCompressor(level=int(name[-1]), threads=0)
                            decompressor = zstd.ZstdDecompressor()
                            encode, decode = compressor.compress, decompressor.decompress
                        started = time.perf_counter()
                        encoded = [encode(shuffle(b) if transform != "none" else b) for b in blocks]
                        compress_seconds = time.perf_counter() - started

                        def restore():
                            return [unshuffle(decode(b)) if transform != "none" else decode(b)
                                    for b in encoded]

                        restored = restore()
                        if restored != blocks:
                            raise RuntimeError("CPU codec failed exact byte comparison")
                        del restored
                        full_times = timed(restore)
                        codec_times = timed(lambda: [decode(b) for b in encoded])
                        # Eight-byte offset per chunk plus the terminal offset.
                        stored = sum(map(len, encoded)) + 8 * (len(encoded) + 1)
                        record = dict(sample_index=row["sample_index"], rank=row["rank"],
                            input_tokens=row["input_tokens"], layers=row["layers"], kind=kind,
                            algorithm=name, transform=transform, block_size=block_size,
                            raw_bytes=len(raw), stored_bytes=stored,
                            stored_fraction=stored / len(raw), factor=len(raw) / stored,
                            bitwise_exact=True, compression_ms=compress_seconds * 1000,
                            restore=summary(full_times, len(raw)),
                            codec_only=summary(codec_times, len(raw)), cpu_threads=1)
                        records.append(record)
                        save(output / "cpu.json", records)
            print("CPU_COMPLETE", row["sample_index"], row["rank"], kind, flush=True)
    save(output / "cpu-complete.json", dict(successful=True, records=len(records)))


def gpu(directory, output):
    import torch
    import nvidia
    nvidia.__path__.append(str(PACKAGES / "nvidia"))
    from nvidia import nvcomp
    output.mkdir(parents=True, exist_ok=True)
    stream = torch.cuda.current_stream()
    records = []

    def gpu_timed(fn, count, repeats=7):
        fn()
        stream.synchronize()
        gpu_seconds, wall_seconds = [], []
        for _ in range(repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            a.record(stream)
            fn()
            b.record(stream)
            b.synchronize()
            wall_seconds.append(time.perf_counter() - started)
            gpu_seconds.append(a.elapsed_time(b) / 1000)
        return dict(cuda=summary(gpu_seconds, count), wall=summary(wall_seconds, count))

    for row, array in inputs(directory):
        if row["rank"] != 0:
            continue
        host = torch.from_numpy(array.copy()).view(torch.uint8).flatten().pin_memory()
        original = host.cuda()
        raw_size = host.numel()
        raw_copy = gpu_timed(lambda: original.copy_(host, non_blocking=True), raw_size)
        for name, algorithm, dtype in [("LZ4", "LZ4", None),
                                       ("ANS_FP16", "ANS", "<f2"),
                                       ("Bitcomp_FP16", "Bitcomp", "<f2"),
                                       ("Zstd", "Zstd", None)]:
            for transform in ["none", "byte_shuffle"]:
                if dtype and transform != "none":
                    continue
                for chunk_size in [65536, 262144]:
                    record = dict(sample_index=row["sample_index"], input_tokens=row["input_tokens"],
                        rank=0, kind="KV", algorithm=name, transform=transform,
                        chunk_size=chunk_size, raw_bytes=raw_size,
                        gpu_name=torch.cuda.get_device_name(), nvcomp_version=nvcomp.__version__)
                    codec = compressed = config = destination_array = None
                    compressed_tensor = compressed_host = None
                    try:
                        source = (original.view(-1, 2).T.contiguous().flatten()
                                  if transform != "none" else original)
                        options = dict(algorithm=algorithm, cuda_stream=stream.cuda_stream,
                                       uncomp_chunk_size=chunk_size)
                        if dtype:
                            options["data_type"] = dtype
                        codec = nvcomp.Codec(**options)
                        started = time.perf_counter()
                        compressed = codec.encode(nvcomp.as_array(source))
                        stream.synchronize()
                        record["compression_wall_ms"] = (time.perf_counter() - started) * 1000
                        started = time.perf_counter()
                        config = codec.decompression_config(compressed)
                        record["header_parse_ms"] = (time.perf_counter() - started) * 1000
                        destination = torch.empty_like(source)
                        restored = torch.empty_like(source)
                        destination_array = nvcomp.as_array(destination)

                        def decode_only():
                            codec.decode(compressed, out=destination_array, decompression_config=config)

                        def restore():
                            decode_only()
                            if transform != "none":
                                restored.view(-1, 2).copy_(destination.view(2, -1).T)

                        restore()
                        stream.synchronize()
                        actual = restored if transform != "none" else destination
                        if not torch.equal(actual, original):
                            raise RuntimeError("GPU codec failed exact byte comparison")
                        # nvCOMP 5.3 DLPack exposes allocation capacity, whereas
                        # buffer_size resolves the valid encoded bitstream size.
                        # Transfer/archive only those valid bytes, not the tail.
                        encoded_size = int(compressed.buffer_size)
                        compressed_tensor = torch.from_dlpack(compressed).view(torch.uint8).flatten()
                        record["compressed_allocation_bytes"] = compressed_tensor.numel()
                        compressed_tensor[encoded_size:].zero_()
                        compressed_tensor = compressed_tensor[:encoded_size]
                        compressed_host = compressed_tensor.cpu().pin_memory()
                        record.update(stored_bytes=compressed_tensor.numel(), bitwise_exact=True,
                                      stored_fraction=compressed_tensor.numel() / raw_size,
                                      factor=raw_size / compressed_tensor.numel())
                        record["decode_only"] = gpu_timed(decode_only, raw_size)
                        record["restore"] = gpu_timed(restore, raw_size)
                        record["raw_H2D"] = raw_copy

                        def upload_restore():
                            compressed_tensor.copy_(compressed_host, non_blocking=True)
                            restore()

                        record["compressed_H2D_restore"] = gpu_timed(upload_restore, raw_size)
                        if not torch.equal(actual, original):
                            raise RuntimeError("Upload plus decode failed exact byte comparison")
                        if name == "ANS_FP16" and chunk_size == 262144:
                            raw_pairs, compressed_pairs = [], []
                            for pair in range(7):
                                operations = [("raw", lambda: original.copy_(host, non_blocking=True)),
                                              ("compressed", upload_restore)]
                                if pair % 2:
                                    operations.reverse()
                                for label, operation in operations:
                                    measured = gpu_timed(operation, raw_size, repeats=1)
                                    target = raw_pairs if label == "raw" else compressed_pairs
                                    target.append(measured["cuda"]["median_ms"])
                            record["paired_upload_cuda_ms"] = dict(raw=raw_pairs,
                                compressed_restore=compressed_pairs,
                                raw_median=float(np.median(raw_pairs)),
                                compressed_restore_median=float(np.median(compressed_pairs)),
                                speedup=float(np.median(raw_pairs) / np.median(compressed_pairs)))
                        record["status"] = "ok"
                    except Exception as exc:
                        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    records.append(record)
                    save(output / "gpu.json", records)
                    print("GPU_CODEC", row["sample_index"], name, transform, chunk_size,
                          record["status"], record.get("error", ""), flush=True)
                    # Native configuration objects depend on the codec manager.
                    # Release them before replacing that manager in the next case.
                    stream.synchronize()
                    destination_array = None
                    compressed_tensor = compressed_host = None
                    config = None
                    compressed = None
                    codec = None
        del original, host
    save(output / "gpu-complete.json", dict(successful=all(r["status"] == "ok" for r in records),
                                            records=len(records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    args = parser.parse_args()
    (cpu if args.mode == "cpu" else gpu)(args.capture.resolve(), args.output.resolve())
