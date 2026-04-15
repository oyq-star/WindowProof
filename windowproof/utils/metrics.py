"""Evaluation metrics for three-way anomaly detection."""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)
from typing import Dict, List, Optional


def three_way_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute metrics for three-way classification.

    Labels: 0=normal, 1=integrity_failure, 2=behavioral_anomaly
    """
    labels = [0, 1, 2]
    label_names = ["normal", "integrity_failure", "behavioral_anomaly"]

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    per_class = {}
    for label, name in zip(labels, label_names):
        y_t_bin = (y_true == label).astype(int)
        y_p_bin = (y_pred == label).astype(int)
        per_class[name] = {
            "precision": float(precision_score(y_t_bin, y_p_bin, zero_division=0)),
            "recall": float(recall_score(y_t_bin, y_p_bin, zero_division=0)),
            "f1": float(f1_score(y_t_bin, y_p_bin, zero_division=0)),
            "support": int(np.sum(y_true == label)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def binary_anomaly_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                           y_pred: Optional[np.ndarray] = None) -> dict:
    """Binary anomaly detection metrics (normal vs anomaly)."""
    if y_pred is None:
        y_pred = (y_scores > 0.5).astype(int)

    result = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) > 1:
        result["auroc"] = float(roc_auc_score(y_true, y_scores))
        result["auprc"] = float(average_precision_score(y_true, y_scores))
    else:
        result["auroc"] = 0.0
        result["auprc"] = 0.0

    return result


def attack_coverage_report(attack_types: List[str], y_true: np.ndarray,
                           y_pred: np.ndarray) -> dict:
    """Report detection performance per attack type."""
    report = {}
    unique_attacks = set(attack_types)

    for attack in unique_attacks:
        mask = np.array([a == attack for a in attack_types])
        if not np.any(mask):
            continue

        sub_true = y_true[mask]
        sub_pred = y_pred[mask]

        detected = np.sum((sub_true > 0) & (sub_pred > 0))
        total = np.sum(sub_true > 0)

        report[attack] = {
            "total": int(total),
            "detected": int(detected),
            "detection_rate": float(detected / max(total, 1)),
            "f1": float(f1_score(sub_true > 0, sub_pred > 0, zero_division=0)),
        }

    return report


def blockchain_metrics_summary(chain_metrics: dict) -> dict:
    """Format blockchain metrics for reporting."""
    return {
        "storage_per_commitment_bytes": (
            chain_metrics["total_storage_bytes"] / max(chain_metrics["total_commitments"], 1)
        ),
        "total_gas_cost": chain_metrics["estimated_gas_cost"],
        "gas_per_trip_avg": (
            chain_metrics["estimated_gas_cost"] /
            max(chain_metrics["total_commitments"], 1)
        ),
        "avg_verification_time_ms": chain_metrics["avg_verification_time_ms"],
        "audit_rate": (
            chain_metrics["total_audits"] /
            max(chain_metrics["total_commitments"], 1)
        ),
        "failure_rate": (
            chain_metrics["total_failures"] /
            max(chain_metrics["total_commitments"] + chain_metrics["total_failures"], 1)
        ),
    }
