"""Checkpoint extraction: extract critical mobility events from raw GPS windows."""

from typing import List, Tuple

from .window_sketch import haversine_distance, compute_bearing


def extract_checkpoints(points: List[Tuple[float, float, float]],
                        stop_speed_threshold: float = 1.0,
                        min_stop_duration: float = 60.0,
                        sharp_turn_threshold: float = 45.0) -> List[int]:
    """Extract indices of critical mobility events (stops, turns, dwells).

    Args:
        points: List of (lat, lon, timestamp) sorted by time.
        stop_speed_threshold: Speed below which a point is considered stopped (m/s).
        min_stop_duration: Minimum duration for a stop event (seconds).
        sharp_turn_threshold: Angle threshold for sharp turns (degrees).

    Returns:
        List of point indices that are checkpoints.
    """
    if len(points) < 2:
        return list(range(len(points)))

    checkpoint_indices = set()

    # Always include first and last points
    checkpoint_indices.add(0)
    checkpoint_indices.add(len(points) - 1)

    # Detect stops
    stop_start = None
    for i in range(1, len(points)):
        d = haversine_distance(points[i - 1][0], points[i - 1][1],
                               points[i][0], points[i][1])
        dt = points[i][2] - points[i - 1][2]
        speed = d / dt if dt > 0 else 0.0

        if speed < stop_speed_threshold:
            if stop_start is None:
                stop_start = i - 1
        else:
            if stop_start is not None:
                stop_duration = points[i - 1][2] - points[stop_start][2]
                if stop_duration >= min_stop_duration:
                    checkpoint_indices.add(stop_start)
                    checkpoint_indices.add(i - 1)
                stop_start = None

    # Handle ongoing stop at end
    if stop_start is not None:
        stop_duration = points[-1][2] - points[stop_start][2]
        if stop_duration >= min_stop_duration:
            checkpoint_indices.add(stop_start)

    # Detect sharp turns
    bearings = []
    for i in range(1, len(points)):
        b = compute_bearing(points[i - 1][0], points[i - 1][1],
                            points[i][0], points[i][1])
        bearings.append(b)

    for i in range(1, len(bearings)):
        diff = abs(bearings[i] - bearings[i - 1])
        if diff > 180:
            diff = 360 - diff
        if diff > sharp_turn_threshold:
            checkpoint_indices.add(i)  # index in points = bearing index

    # Detect speed anomalies (sudden acceleration/deceleration)
    for i in range(1, len(points)):
        d = haversine_distance(points[i - 1][0], points[i - 1][1],
                               points[i][0], points[i][1])
        dt = points[i][2] - points[i - 1][2]
        if dt > 0:
            speed = d / dt
            if speed > 50.0:  # > 180 km/h, suspicious
                checkpoint_indices.add(i)

    return sorted(checkpoint_indices)


def compute_checkpoint_features(points: List[Tuple[float, float, float]],
                                checkpoint_indices: List[int]) -> dict:
    """Compute features from sparse checkpoint representation.

    Returns features describing inter-checkpoint behavior.
    """
    if len(checkpoint_indices) < 2:
        return {
            "num_checkpoints": len(checkpoint_indices),
            "checkpoint_density": 0.0,
            "mean_inter_checkpoint_time": 0.0,
            "std_inter_checkpoint_time": 0.0,
            "mean_inter_checkpoint_dist": 0.0,
            "std_inter_checkpoint_dist": 0.0,
            "max_inter_checkpoint_speed": 0.0,
            "sequence_regularity": 0.0,
            "missing_ratio": 0.0,
        }

    checkpoints = [points[i] for i in checkpoint_indices]
    total_points = len(points)

    inter_times = []
    inter_dists = []
    inter_speeds = []
    for i in range(1, len(checkpoints)):
        dt = checkpoints[i][2] - checkpoints[i - 1][2]
        d = haversine_distance(checkpoints[i - 1][0], checkpoints[i - 1][1],
                               checkpoints[i][0], checkpoints[i][1])
        inter_times.append(dt)
        inter_dists.append(d)
        if dt > 0:
            inter_speeds.append(d / dt)

    import numpy as np
    inter_times = np.array(inter_times)
    inter_dists = np.array(inter_dists)

    # Sequence regularity: how evenly spaced are checkpoints in time?
    if len(inter_times) > 1 and np.mean(inter_times) > 0:
        regularity = 1.0 - (np.std(inter_times) / np.mean(inter_times))
        regularity = max(0.0, regularity)
    else:
        regularity = 1.0

    # Missing ratio: fraction of points between checkpoints that are not checkpoints
    missing_ratio = 1.0 - len(checkpoint_indices) / max(total_points, 1)

    return {
        "num_checkpoints": len(checkpoint_indices),
        "checkpoint_density": len(checkpoint_indices) / max(total_points, 1),
        "mean_inter_checkpoint_time": float(np.mean(inter_times)),
        "std_inter_checkpoint_time": float(np.std(inter_times)),
        "mean_inter_checkpoint_dist": float(np.mean(inter_dists)),
        "std_inter_checkpoint_dist": float(np.std(inter_dists)),
        "max_inter_checkpoint_speed": max(inter_speeds) if inter_speeds else 0.0,
        "sequence_regularity": regularity,
        "missing_ratio": missing_ratio,
    }
