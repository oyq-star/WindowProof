"""Window sketch computation for raw trajectory windows."""

import math
from collections import Counter
from typing import List, Tuple

import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute bearing in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def encode_geohash_simple(lat: float, lon: float, precision: int = 6) -> str:
    """Simple geohash encoding for cell bitmap."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    chars = "0123456789bcdefghjkmnpqrstuvwxyz"
    geohash = []
    is_lon = True
    bit = 0
    ch = 0
    for _ in range(precision * 5):
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                ch = ch * 2 + 1
                lon_range[0] = mid
            else:
                ch = ch * 2
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                ch = ch * 2 + 1
                lat_range[0] = mid
            else:
                ch = ch * 2
                lat_range[1] = mid
        is_lon = not is_lon
        bit += 1
        if bit == 5:
            geohash.append(chars[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


def compute_window_sketch(points: List[Tuple[float, float, float]],
                          geohash_precision: int = 6) -> dict:
    """Compute compact sketch S_t for a trajectory window.

    Args:
        points: List of (lat, lon, timestamp) tuples, sorted by timestamp.
        geohash_precision: Precision for geohash cell bitmap.

    Returns:
        Dictionary with sketch features.
    """
    n = len(points)
    if n == 0:
        return _empty_sketch()

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    timestamps = [p[2] for p in points]

    duration = timestamps[-1] - timestamps[0] if n > 1 else 0.0

    # Distances and speeds
    distances = []
    speeds = []
    bearings = []
    for i in range(1, n):
        d = haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
        dt = timestamps[i] - timestamps[i - 1]
        distances.append(d)
        if dt > 0:
            speeds.append(d / dt)
        bearings.append(compute_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i]))

    total_path_length = sum(distances)
    max_speed = max(speeds) if speeds else 0.0
    mean_speed = np.mean(speeds) if speeds else 0.0

    # Stop ratio: fraction of time spent nearly stationary
    stop_count = sum(1 for s in speeds if s < 1.0)
    stop_ratio = stop_count / max(len(speeds), 1)

    # Turn count: number of sharp heading changes
    turn_count = 0
    for i in range(1, len(bearings)):
        diff = abs(bearings[i] - bearings[i - 1])
        if diff > 180:
            diff = 360 - diff
        if diff > 45:
            turn_count += 1

    # Heading entropy
    if bearings:
        binned = [int(b // 45) for b in bearings]
        counts = Counter(binned)
        total_b = sum(counts.values())
        probs = [c / total_b for c in counts.values()]
        heading_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        heading_entropy = 0.0

    # Geohash cell bitmap
    geohashes = set()
    for lat, lon, _ in points:
        geohashes.add(encode_geohash_simple(lat, lon, geohash_precision))
    cell_count = len(geohashes)

    return {
        "sample_count": n,
        "duration_sec": duration,
        "total_path_length_m": total_path_length,
        "max_speed_mps": max_speed,
        "mean_speed_mps": float(mean_speed),
        "stop_ratio": stop_ratio,
        "turn_count": turn_count,
        "heading_entropy": heading_entropy,
        "cell_count": cell_count,
        "geohash_set": sorted(geohashes),
    }


def sketch_to_vector(sketch: dict) -> np.ndarray:
    """Convert sketch dict to numeric feature vector (excluding geohash set)."""
    return np.array([
        sketch["sample_count"],
        sketch["duration_sec"],
        sketch["total_path_length_m"],
        sketch["max_speed_mps"],
        sketch["mean_speed_mps"],
        sketch["stop_ratio"],
        sketch["turn_count"],
        sketch["heading_entropy"],
        sketch["cell_count"],
    ], dtype=np.float64)


def _empty_sketch() -> dict:
    return {
        "sample_count": 0,
        "duration_sec": 0.0,
        "total_path_length_m": 0.0,
        "max_speed_mps": 0.0,
        "mean_speed_mps": 0.0,
        "stop_ratio": 0.0,
        "turn_count": 0,
        "heading_entropy": 0.0,
        "cell_count": 0,
        "geohash_set": [],
    }
