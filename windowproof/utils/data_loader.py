"""Data loaders for public trajectory datasets."""

import os
import csv
import random
from typing import List, Tuple, Optional

import numpy as np


def load_porto_taxi(data_path: str, max_trajectories: int = 5000,
                    min_length: int = 10, seed: int = 42) -> List[List[Tuple]]:
    """Load Porto taxi dataset (train.csv from Kaggle).

    Expected CSV columns: TRIP_ID, CALL_TYPE, ..., POLYLINE
    POLYLINE is a JSON list of [lon, lat] pairs.

    Returns list of trajectories, each as [(lat, lon, timestamp), ...].
    """
    import json

    trajectories = []
    rng = random.Random(seed)

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

        if len(polyline) < min_length:
            continue

        # Porto data: 15-second intervals, POLYLINE is [[lon, lat], ...]
        base_timestamp = float(row.get("TIMESTAMP", 0))
        traj = []
        for i, (lon, lat) in enumerate(polyline):
            traj.append((float(lat), float(lon), base_timestamp + i * 15.0))

        trajectories.append(traj)

    return trajectories


def load_tdrive(data_dir: str, max_trajectories: int = 5000,
                min_length: int = 10, seed: int = 42) -> List[List[Tuple]]:
    """Load T-Drive Beijing taxi dataset.

    Expected format: one file per taxi, each line: taxi_id, datetime, lon, lat
    """
    from datetime import datetime

    trajectories = []
    rng = random.Random(seed)

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    rng.shuffle(txt_files)

    for fname in txt_files:
        if len(trajectories) >= max_trajectories:
            break

        filepath = os.path.join(data_dir, fname)
        traj = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    dt = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
                    lon = float(parts[2].strip())
                    lat = float(parts[3].strip())
                    ts = dt.timestamp()
                    traj.append((lat, lon, ts))
                except (ValueError, IndexError):
                    continue

        if len(traj) >= min_length:
            traj.sort(key=lambda p: p[2])
            trajectories.append(traj)

    return trajectories


def generate_synthetic_trajectories(n_trajectories: int = 1000,
                                    min_length: int = 20,
                                    max_length: int = 100,
                                    center_lat: float = 41.15,
                                    center_lon: float = -8.61,
                                    seed: int = 42) -> List[List[Tuple]]:
    """Generate synthetic taxi-like trajectories for testing.

    Simulates random walks with realistic speed and turn patterns.
    Center defaults to Porto, Portugal.
    """
    rng = np.random.RandomState(seed)
    trajectories = []

    for _ in range(n_trajectories):
        n_points = rng.randint(min_length, max_length + 1)
        lat = center_lat + rng.uniform(-0.05, 0.05)
        lon = center_lon + rng.uniform(-0.05, 0.05)
        timestamp = rng.uniform(1e9, 1.1e9)
        bearing = rng.uniform(0, 360)

        traj = [(lat, lon, timestamp)]

        for _ in range(n_points - 1):
            speed = rng.uniform(2, 15)  # m/s (7-54 km/h)
            dt = 15.0  # seconds
            bearing += rng.normal(0, 15)
            bearing %= 360

            d_lat = speed * dt * np.cos(np.radians(bearing)) / 111000.0
            d_lon = speed * dt * np.sin(np.radians(bearing)) / (
                111000.0 * np.cos(np.radians(lat))
            )
            lat += d_lat
            lon += d_lon
            timestamp += dt

            # Occasional stops
            if rng.random() < 0.05:
                timestamp += rng.uniform(30, 120)

            traj.append((lat, lon, timestamp))

        trajectories.append(traj)

    return trajectories
