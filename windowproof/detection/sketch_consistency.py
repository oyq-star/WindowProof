"""Window-sketch consistency detection: the core algorithmic novelty."""

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from typing import List, Tuple

from ..capture.window_sketch import compute_window_sketch, sketch_to_vector
from ..capture.checkpoint_extractor import compute_checkpoint_features


def reconstruct_sketch_from_checkpoints(checkpoints: List[Tuple[float, float, float]],
                                        geohash_precision: int = 6) -> dict:
    """Reconstruct an estimated sketch from sparse checkpoints only.
    This is the 'untrusted' view that should be consistent with the committed sketch."""
    return compute_window_sketch(checkpoints, geohash_precision)


def compute_sketch_residual(committed_sketch: dict, reconstructed_sketch: dict,
                            metric: str = "cosine") -> dict:
    """Compute residual between committed raw-window sketch and checkpoint-reconstructed sketch.

    Large residual indicates the sparse view is inconsistent with the committed raw window,
    suggesting omission or tampering.
    """
    s_committed = sketch_to_vector(committed_sketch)
    s_reconstructed = sketch_to_vector(reconstructed_sketch)

    # Normalize to prevent scale issues
    norm_c = np.linalg.norm(s_committed)
    norm_r = np.linalg.norm(s_reconstructed)

    if norm_c == 0 and norm_r == 0:
        return {"residual_score": 0.0, "per_feature_residual": np.zeros(len(s_committed)),
                "metric": metric}

    if metric == "cosine":
        if norm_c == 0 or norm_r == 0:
            score = 1.0
        else:
            score = cosine_dist(s_committed, s_reconstructed)
    elif metric == "euclidean":
        safe_c = s_committed / (norm_c + 1e-10)
        safe_r = s_reconstructed / (norm_r + 1e-10)
        score = float(np.linalg.norm(safe_c - safe_r))
    else:
        score = cosine_dist(s_committed, s_reconstructed) if norm_c > 0 and norm_r > 0 else 1.0

    # Per-feature relative residual
    per_feature = np.abs(s_committed - s_reconstructed) / (np.abs(s_committed) + 1e-10)

    # Geohash coverage residual
    committed_cells = set(committed_sketch.get("geohash_set", []))
    reconstructed_cells = set(reconstructed_sketch.get("geohash_set", []))
    if committed_cells:
        cell_coverage = len(committed_cells & reconstructed_cells) / len(committed_cells)
    else:
        cell_coverage = 1.0

    return {
        "residual_score": float(score),
        "cell_coverage": cell_coverage,
        "per_feature_residual": per_feature,
        "metric": metric,
    }


def build_detection_features(checkpoint_features: dict,
                             committed_sketch: dict,
                             residual_info: dict,
                             integrity_flags: dict) -> np.ndarray:
    """Build the full feature vector for the three-way detector.

    Combines:
    - Checkpoint-derived features (sparse behavioral)
    - Committed sketch features (trusted aggregate)
    - Sketch residual (consistency signal)
    - Integrity flags (provenance signals)
    """
    # Checkpoint features
    cp_feats = [
        checkpoint_features["num_checkpoints"],
        checkpoint_features["checkpoint_density"],
        checkpoint_features["mean_inter_checkpoint_time"],
        checkpoint_features["std_inter_checkpoint_time"],
        checkpoint_features["mean_inter_checkpoint_dist"],
        checkpoint_features["std_inter_checkpoint_dist"],
        checkpoint_features["max_inter_checkpoint_speed"],
        checkpoint_features["sequence_regularity"],
        checkpoint_features["missing_ratio"],
    ]

    # Sketch features (from trusted committed sketch)
    sk_feats = [
        committed_sketch["sample_count"],
        committed_sketch["duration_sec"],
        committed_sketch["total_path_length_m"],
        committed_sketch["max_speed_mps"],
        committed_sketch["mean_speed_mps"],
        committed_sketch["stop_ratio"],
        committed_sketch["turn_count"],
        committed_sketch["heading_entropy"],
        committed_sketch["cell_count"],
    ]

    # Residual features
    res_feats = [
        residual_info["residual_score"],
        residual_info["cell_coverage"],
    ]

    # Integrity flags
    int_feats = [
        float(integrity_flags.get("window_missing", False)),
        float(integrity_flags.get("late_commitment", False)),
        float(integrity_flags.get("proof_failed", False)),
        float(integrity_flags.get("density_violation", False)),
    ]

    return np.array(cp_feats + sk_feats + res_feats + int_feats, dtype=np.float64)


FEATURE_NAMES = [
    "num_checkpoints", "checkpoint_density",
    "mean_inter_cp_time", "std_inter_cp_time",
    "mean_inter_cp_dist", "std_inter_cp_dist",
    "max_inter_cp_speed", "sequence_regularity", "missing_ratio",
    "sample_count", "duration_sec", "total_path_length_m",
    "max_speed_mps", "mean_speed_mps", "stop_ratio",
    "turn_count", "heading_entropy", "cell_count",
    "sketch_residual", "cell_coverage",
    "flag_window_missing", "flag_late_commit",
    "flag_proof_failed", "flag_density_violation",
]
