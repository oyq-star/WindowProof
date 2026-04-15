"""Loaders for real trajectory datasets (preprocessed CSV format)."""

import csv
import random
from collections import defaultdict
from typing import List, Tuple
from pathlib import Path


def load_preprocessed_csv(data_path: str, max_trajectories: int = 2000,
                          min_length: int = 10, seed: int = 42) -> List[List[Tuple]]:
    """Load trajectories from preprocessed CSV format.

    Expected columns: traj_id, point_idx, lat, lon, timestamp
    Returns list of trajectories as [(lat, lon, timestamp), ...].
    """
    rng = random.Random(seed)
    traj_dict = defaultdict(list)

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["traj_id"]
            lat = float(row["lat"])
            lon = float(row["lon"])
            ts = float(row["timestamp"])
            traj_dict[tid].append((lat, lon, ts))

    # Sort each trajectory by timestamp
    trajectories = []
    for tid in traj_dict:
        traj = sorted(traj_dict[tid], key=lambda x: x[2])
        if len(traj) >= min_length:
            trajectories.append(traj)

    rng.shuffle(trajectories)
    return trajectories[:max_trajectories]


def load_porto_raw(data_path: str, max_trajectories: int = 2000,
                   min_length: int = 10, seed: int = 42) -> List[List[Tuple]]:
    """Load Porto taxi dataset directly from train.csv.

    Porto format: TRIP_ID,CALL_TYPE,...,TIMESTAMP,...,POLYLINE
    POLYLINE is JSON [[lon,lat], ...] with 15s intervals.
    """
    import json

    rng = random.Random(seed)
    trajectories = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rng.shuffle(rows)

    for row in rows:
        if len(trajectories) >= max_trajectories:
            break
        try:
            polyline = json.loads(row.get("POLYLINE", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        if not polyline or len(polyline) < min_length:
            continue

        # Filter out empty polylines and clearly bad data
        if polyline == []:
            continue

        base_ts = float(row.get("TIMESTAMP", 0))
        traj = []
        valid = True
        for i, point in enumerate(polyline):
            if len(point) != 2:
                valid = False
                break
            lon, lat = float(point[0]), float(point[1])
            # Basic sanity: Porto area
            if not (-9.5 < lon < -7.5 and 40.5 < lat < 41.5):
                valid = False
                break
            traj.append((lat, lon, base_ts + i * 15.0))

        if valid and len(traj) >= min_length:
            trajectories.append(traj)

    return trajectories


def load_tdrive_raw(data_dir: str, max_trajectories: int = 2000,
                    min_length: int = 10, seed: int = 42) -> List[List[Tuple]]:
    """Load T-Drive dataset directly from raw txt files.

    T-Drive format per line: taxi_id, datetime, longitude, latitude
    """
    from datetime import datetime
    import os

    rng = random.Random(seed)
    all_trajectories = []

    # Find all txt files
    txt_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))

    rng.shuffle(txt_files)

    for fpath in txt_files:
        if len(all_trajectories) >= max_trajectories:
            break

        points = []
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    dt = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
                    lon = float(parts[2].strip())
                    lat = float(parts[3].strip())
                    if 39.4 < lat < 41.0 and 115.5 < lon < 117.5:
                        points.append((lat, lon, dt.timestamp()))
                except (ValueError, IndexError):
                    continue

        if not points:
            continue

        points.sort(key=lambda x: x[2])

        # Split by time gaps > 30 minutes
        trips = []
        current = [points[0]]
        for i in range(1, len(points)):
            if points[i][2] - points[i - 1][2] > 1800:
                if len(current) >= min_length:
                    trips.append(current)
                current = [points[i]]
            else:
                current.append(points[i])
        if len(current) >= min_length:
            trips.append(current)

        all_trajectories.extend(trips)

    rng_final = random.Random(seed)
    rng_final.shuffle(all_trajectories)
    return all_trajectories[:max_trajectories]
