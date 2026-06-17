"""Scalability microbenchmark for the WindowChain local simulator.

This benchmark measures local, single-threaded simulator latency for the
commit/disclose/verify path. It does not emulate consensus, networking, or a
real blockchain VM; those remain testnet validation tasks.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windowproof.blockchain.smart_contract import WindowChainContract
from windowproof.blockchain.window_chain import WindowChain
from windowproof.capture.window_sketch import compute_window_sketch


def percentile(values, pct):
    """Return percentile using linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(values):
    return {
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": statistics.median(values) if values else 0.0,
        "p95_ms": percentile(values, 95),
        "max_ms": max(values) if values else 0.0,
    }


def make_window(device_index, window_index, points_per_window):
    """Create a deterministic pseudo-trajectory window."""
    base_lat = 39.90 + 0.0001 * (device_index % 100)
    base_lon = 116.30 + 0.0001 * (device_index // 100)
    start_ts = 1_700_000_000 + 300 * window_index
    points = []
    for i in range(points_per_window):
        lat = base_lat + 0.00005 * i + 0.00001 * math.sin(i + device_index)
        lon = base_lon + 0.00004 * i + 0.00001 * math.cos(i + window_index)
        ts = start_ts + 15 * i
        points.append((lat, lon, ts))
    return points


def run_microbenchmark(num_devices, windows_per_device, points_per_window, checkpoint_stride):
    contract = WindowChainContract(min_checkpoint_density=0.1)
    chain = WindowChain(contract)

    commit_ms = []
    disclose_ms = []
    verify_ms = []
    storage_bytes = []
    total_checkpoints = 0

    t0 = time.perf_counter()
    for device_index in range(num_devices):
        device_id = f"device-{device_index:05d}"
        for window_id in range(windows_per_device):
            points = make_window(device_index, window_id, points_per_window)
            sketch = compute_window_sketch(points)

            start = time.perf_counter()
            commit = chain.commit_window(device_id, window_id, points, sketch)
            commit_ms.append((time.perf_counter() - start) * 1000.0)
            if not commit["success"]:
                raise RuntimeError(f"commit failed: {commit}")
            storage_bytes.append(commit["storage_bytes"])

            checkpoint_indices = list(range(0, points_per_window, checkpoint_stride))
            if checkpoint_indices[-1] != points_per_window - 1:
                checkpoint_indices.append(points_per_window - 1)

            start = time.perf_counter()
            disclosure = chain.disclose_checkpoints(
                device_id=device_id,
                window_id=window_id,
                checkpoint_indices=checkpoint_indices,
                raw_points=points,
                merkle_tree=commit["merkle_tree"],
                total_points=points_per_window,
            )
            disclose_ms.append((time.perf_counter() - start) * 1000.0)
            if not disclosure["success"]:
                raise RuntimeError(f"disclosure failed: {disclosure}")

            checkpoints = contract.disclosures[device_id][window_id].checkpoints
            proofs = contract.disclosures[device_id][window_id].checkpoint_proofs
            start = time.perf_counter()
            for checkpoint, proof in zip(checkpoints, proofs):
                ok = chain.verify_checkpoint(checkpoint, proof, commit["merkle_root"])
                if not ok:
                    raise RuntimeError("checkpoint proof verification failed")
            verify_ms.append((time.perf_counter() - start) * 1000.0)
            total_checkpoints += len(checkpoints)

    total_runtime_s = time.perf_counter() - t0
    total_windows = num_devices * windows_per_device

    return {
        "benchmark_scope": (
            "local single-threaded WindowChain simulator; excludes consensus, "
            "networking, mempool delay, and real blockchain VM execution"
        ),
        "num_devices": num_devices,
        "windows_per_device": windows_per_device,
        "total_windows": total_windows,
        "points_per_window": points_per_window,
        "avg_checkpoints_per_window": total_checkpoints / total_windows,
        "total_runtime_s": total_runtime_s,
        "pipeline_windows_per_second": total_windows / total_runtime_s,
        "commit_latency": summarize_ms(commit_ms),
        "disclose_latency": summarize_ms(disclose_ms),
        "checkpoint_verify_latency": summarize_ms(verify_ms),
        "storage_bytes_per_window": {
            "mean": statistics.fmean(storage_bytes),
            "min": min(storage_bytes),
            "max": max(storage_bytes),
        },
        "contract_metrics": contract.get_metrics(),
    }


def fleet_projection(batch_sizes=(1, 10, 100)):
    device_counts = [1_000, 10_000, 100_000]
    window_lengths = [60, 300, 600]
    rows = []
    for devices in device_counts:
        for tau in window_lengths:
            raw_commit_tps = devices / tau
            row = {
                "devices": devices,
                "window_sec": tau,
                "raw_commit_tps": raw_commit_tps,
                "batched_commit_tps": {
                    str(batch): raw_commit_tps / batch for batch in batch_sizes
                },
            }
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices", type=int, default=100)
    parser.add_argument("--windows-per-device", type=int, default=50)
    parser.add_argument("--points-per-window", type=int, default=20)
    parser.add_argument("--checkpoint-stride", type=int, default=5)
    parser.add_argument("--output", default="code/results/scalability_microbenchmark.json")
    args = parser.parse_args()

    results = run_microbenchmark(
        num_devices=args.num_devices,
        windows_per_device=args.windows_per_device,
        points_per_window=args.points_per_window,
        checkpoint_stride=args.checkpoint_stride,
    )
    results["fleet_projection"] = fleet_projection()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
