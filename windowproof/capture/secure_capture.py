"""Secure capture module simulation (TEE-based raw window commitment)."""

import hashlib
import hmac
from typing import List, Tuple

from .window_sketch import compute_window_sketch
from .checkpoint_extractor import extract_checkpoints


class SecureCaptureModule:
    """Simulates a trusted execution environment that captures raw GPS,
    computes window sketches, and signs commitments before any untrusted processing."""

    def __init__(self, device_id: str, signing_key: bytes = b"secure_key",
                 window_size_sec: float = 300.0,
                 geohash_precision: int = 6):
        self.device_id = device_id
        self.signing_key = signing_key
        self.window_size_sec = window_size_sec
        self.geohash_precision = geohash_precision
        self.window_counter = 0

    def process_trajectory(self, raw_points: List[Tuple[float, float, float]]) -> List[dict]:
        """Split a trajectory into time windows and process each.

        Args:
            raw_points: Full trajectory as [(lat, lon, timestamp), ...] sorted by time.

        Returns:
            List of window records with raw points, sketch, checkpoints, etc.
        """
        if not raw_points:
            return []

        windows = self._split_into_windows(raw_points)
        results = []

        for window_points in windows:
            record = self._process_window(window_points)
            results.append(record)

        return results

    def _split_into_windows(self, points: List[Tuple[float, float, float]]) -> List[List[Tuple]]:
        """Split trajectory into fixed-duration time windows."""
        if not points:
            return []

        windows = []
        start_time = points[0][2]
        current_window = []

        for p in points:
            if p[2] - start_time >= self.window_size_sec and current_window:
                windows.append(current_window)
                current_window = [p]
                start_time = p[2]
            else:
                current_window.append(p)

        if current_window:
            windows.append(current_window)

        return windows

    def _process_window(self, points: List[Tuple[float, float, float]]) -> dict:
        """Process a single time window: compute sketch, extract checkpoints, sign."""
        window_id = self.window_counter
        self.window_counter += 1

        sketch = compute_window_sketch(points, self.geohash_precision)

        checkpoint_indices = extract_checkpoints(points)

        sketch_str = str(sorted(
            {k: v for k, v in sketch.items() if k != "geohash_set"}.items()
        ))
        sketch_hash = hashlib.sha256(sketch_str.encode()).digest()

        payload = f"{self.device_id}:{window_id}:{sketch_hash.hex()}".encode()
        signature = hmac.new(self.signing_key, payload, hashlib.sha256).digest()

        return {
            "device_id": self.device_id,
            "window_id": window_id,
            "raw_points": points,
            "sketch": sketch,
            "sketch_hash": sketch_hash,
            "checkpoint_indices": checkpoint_indices,
            "signature": signature,
            "point_count": len(points),
        }
