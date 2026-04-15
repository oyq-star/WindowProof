"""Attack simulator: inject both behavioral anomalies and integrity attacks."""

import random
import copy
import numpy as np
from typing import List, Tuple

from ..capture.window_sketch import haversine_distance


class AttackSimulator:
    """Generate synthetic attacks on trajectory data."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    # ========================
    # Behavioral Anomalies
    # ========================

    def inject_detour(self, points: List[Tuple], detour_ratio: float = 0.3,
                      offset_m: float = 500.0) -> Tuple[List[Tuple], dict]:
        """Inject a detour by shifting a segment of points laterally."""
        pts = list(points)
        n = len(pts)
        seg_len = max(2, int(n * detour_ratio))
        start = self.rng.randint(0, max(0, n - seg_len))

        offset_lat = offset_m / 111000.0
        offset_lon = offset_m / (111000.0 * np.cos(np.radians(pts[start][0])))
        direction = self.rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])

        for i in range(start, min(start + seg_len, n)):
            pts[i] = (
                pts[i][0] + direction[0] * offset_lat,
                pts[i][1] + direction[1] * offset_lon,
                pts[i][2],
            )
        return pts, {"type": "detour", "start": start, "length": seg_len}

    def inject_loop(self, points: List[Tuple]) -> Tuple[List[Tuple], dict]:
        """Inject a circular loop in the middle of trajectory."""
        pts = list(points)
        n = len(pts)
        if n < 10:
            return pts, {"type": "loop", "injected": False}

        center_idx = n // 2
        center = pts[center_idx]
        radius_lat = 200.0 / 111000.0
        radius_lon = 200.0 / (111000.0 * np.cos(np.radians(center[0])))
        num_loop_pts = 8
        base_time = center[2]

        loop_pts = []
        for i in range(num_loop_pts):
            angle = 2 * np.pi * i / num_loop_pts
            lat = center[0] + radius_lat * np.cos(angle)
            lon = center[1] + radius_lon * np.sin(angle)
            t = base_time + i * 15.0
            loop_pts.append((lat, lon, t))

        # Shift subsequent timestamps
        time_shift = num_loop_pts * 15.0
        new_pts = pts[:center_idx]
        new_pts.extend(loop_pts)
        for p in pts[center_idx:]:
            new_pts.append((p[0], p[1], p[2] + time_shift))

        return new_pts, {"type": "loop", "injected": True, "center": center_idx}

    def inject_abnormal_stop(self, points: List[Tuple],
                             duration: float = 600.0) -> Tuple[List[Tuple], dict]:
        """Inject an abnormally long stop."""
        pts = list(points)
        n = len(pts)
        if n < 4:
            return pts, {"type": "abnormal_stop", "injected": False}

        stop_idx = self.rng.randint(1, n - 2)
        stop_pt = pts[stop_idx]
        num_stop_pts = int(duration / 30.0)

        stop_pts = []
        for i in range(num_stop_pts):
            jitter_lat = self.np_rng.normal(0, 0.00001)
            jitter_lon = self.np_rng.normal(0, 0.00001)
            t = stop_pt[2] + i * 30.0
            stop_pts.append((stop_pt[0] + jitter_lat, stop_pt[1] + jitter_lon, t))

        new_pts = pts[:stop_idx]
        new_pts.extend(stop_pts)
        for p in pts[stop_idx:]:
            new_pts.append((p[0], p[1], p[2] + duration))

        return new_pts, {"type": "abnormal_stop", "duration": duration}

    def inject_speed_burst(self, points: List[Tuple],
                           factor: float = 3.0) -> Tuple[List[Tuple], dict]:
        """Compress timestamps in a segment to simulate impossible speed."""
        pts = list(points)
        n = len(pts)
        seg_len = max(2, n // 5)
        start = self.rng.randint(0, max(0, n - seg_len))

        base_time = pts[start][2]
        for i in range(start, min(start + seg_len, n)):
            original_offset = pts[i][2] - base_time
            pts[i] = (pts[i][0], pts[i][1], base_time + original_offset / factor)

        # Adjust subsequent points
        time_saved = (pts[min(start + seg_len, n) - 1][2] - base_time) * (1 - 1 / factor)
        for i in range(min(start + seg_len, n), n):
            pts[i] = (pts[i][0], pts[i][1], pts[i][2] - time_saved)

        return pts, {"type": "speed_burst", "factor": factor, "start": start}

    def inject_teleport(self, points: List[Tuple],
                        distance_m: float = 5000.0) -> Tuple[List[Tuple], dict]:
        """Inject a teleportation jump."""
        pts = list(points)
        n = len(pts)
        if n < 3:
            return pts, {"type": "teleport", "injected": False}

        jump_idx = self.rng.randint(1, n - 1)
        offset_lat = distance_m / 111000.0
        angle = self.rng.uniform(0, 2 * np.pi)

        for i in range(jump_idx, n):
            pts[i] = (
                pts[i][0] + offset_lat * np.cos(angle),
                pts[i][1] + offset_lat * np.sin(angle),
                pts[i][2],
            )
        return pts, {"type": "teleport", "jump_idx": jump_idx, "distance_m": distance_m}

    # ========================
    # Integrity Attacks
    # ========================

    def attack_point_deletion(self, points: List[Tuple],
                              ratio: float = 0.3) -> Tuple[List[Tuple], dict]:
        """Delete a fraction of points (simulating selective omission)."""
        pts = list(points)
        n_delete = max(1, int(len(pts) * ratio))
        indices = sorted(self.rng.sample(range(1, len(pts) - 1), min(n_delete, len(pts) - 2)))
        remaining = [p for i, p in enumerate(pts) if i not in set(indices)]
        return remaining, {"type": "point_deletion", "deleted_count": len(indices)}

    def attack_point_injection(self, points: List[Tuple],
                               count: int = 10) -> Tuple[List[Tuple], dict]:
        """Inject fake GPS points into the trajectory."""
        pts = list(points)
        for _ in range(count):
            idx = self.rng.randint(0, len(pts) - 1)
            base = pts[idx]
            fake_lat = base[0] + self.np_rng.normal(0, 0.005)
            fake_lon = base[1] + self.np_rng.normal(0, 0.005)
            fake_t = base[2] + self.rng.uniform(-30, 30)
            pts.append((fake_lat, fake_lon, fake_t))

        pts.sort(key=lambda p: p[2])
        return pts, {"type": "point_injection", "injected_count": count}

    def attack_timestamp_shift(self, points: List[Tuple],
                               shift_sec: float = 120.0) -> Tuple[List[Tuple], dict]:
        """Shift timestamps of a segment."""
        pts = list(points)
        n = len(pts)
        seg_len = max(2, n // 4)
        start = self.rng.randint(0, max(0, n - seg_len))

        for i in range(start, min(start + seg_len, n)):
            pts[i] = (pts[i][0], pts[i][1], pts[i][2] + shift_sec)

        pts.sort(key=lambda p: p[2])
        return pts, {"type": "timestamp_shift", "shift_sec": shift_sec}

    def attack_replay(self, points: List[Tuple],
                      segment_length: int = 20) -> Tuple[List[Tuple], dict]:
        """Replay a past segment to mask current trajectory."""
        pts = list(points)
        n = len(pts)
        if n < segment_length * 2:
            return pts, {"type": "replay", "injected": False}

        src_start = self.rng.randint(0, n // 2)
        replay_seg = pts[src_start:src_start + segment_length]
        dst_start = self.rng.randint(n // 2, max(n // 2, n - segment_length))

        time_offset = pts[dst_start][2] - replay_seg[0][2]
        for i, seg_pt in enumerate(replay_seg):
            target_idx = dst_start + i
            if target_idx < n:
                pts[target_idx] = (seg_pt[0], seg_pt[1], seg_pt[2] + time_offset)

        return pts, {"type": "replay", "src_start": src_start, "dst_start": dst_start}

    def attack_window_drop(self, windows: List[dict],
                           drop_prob: float = 0.1) -> Tuple[List[dict], List[int]]:
        """Drop entire windows (simulating window omission attack)."""
        kept = []
        dropped_ids = []
        for w in windows:
            if self.rng.random() > drop_prob:
                kept.append(w)
            else:
                dropped_ids.append(w["window_id"])
        return kept, dropped_ids

    # ========================
    # Batch generation
    # ========================

    def generate_attack(self, points: List[Tuple],
                        attack_type: str, **kwargs) -> Tuple[List[Tuple], dict]:
        """Generate a specific attack type."""
        attack_map = {
            "detour": self.inject_detour,
            "loop": self.inject_loop,
            "abnormal_stop": self.inject_abnormal_stop,
            "speed_burst": self.inject_speed_burst,
            "teleport": self.inject_teleport,
            "point_deletion": self.attack_point_deletion,
            "point_injection": self.attack_point_injection,
            "timestamp_shift": self.attack_timestamp_shift,
            "replay": self.attack_replay,
        }
        if attack_type not in attack_map:
            raise ValueError(f"Unknown attack type: {attack_type}")
        return attack_map[attack_type](points, **kwargs)

    def generate_random_attack(self, points: List[Tuple],
                               category: str = "both") -> Tuple[List[Tuple], dict]:
        """Generate a random attack from the specified category."""
        behavioral = ["detour", "loop", "abnormal_stop", "speed_burst", "teleport"]
        integrity = ["point_deletion", "point_injection", "timestamp_shift", "replay"]

        if category == "behavioral":
            attack_type = self.rng.choice(behavioral)
        elif category == "integrity":
            attack_type = self.rng.choice(integrity)
        else:
            attack_type = self.rng.choice(behavioral + integrity)

        return self.generate_attack(points, attack_type)
