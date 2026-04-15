"""Three-way trustworthy decision layer: provenance failure / behavioral anomaly / normal."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


class ThreeWayDetector:
    """WindowProof three-way anomaly detector.

    Outputs one of three decisions per window:
    - INTEGRITY_FAILURE: provenance/data integrity issue detected
    - BEHAVIORAL_ANOMALY: genuine mobility anomaly on verified data
    - NORMAL: no anomaly detected
    """

    INTEGRITY_FAILURE = "integrity_failure"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    NORMAL = "normal"

    def __init__(self,
                 integrity_threshold: float = 0.3,
                 behavior_contamination: float = 0.05,
                 n_estimators: int = 100,
                 audit_threshold: float = 0.5,
                 random_state: int = 42):
        self.integrity_threshold = integrity_threshold
        self.audit_threshold = audit_threshold
        self.random_state = random_state

        self.behavior_detector = IsolationForest(
            n_estimators=n_estimators,
            contamination=behavior_contamination,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Feature indices
        self._integrity_feature_indices = list(range(18, 24))  # residual + flags
        self._behavior_feature_indices = list(range(0, 18))    # checkpoint + sketch

    def fit(self, X: np.ndarray):
        """Fit on normal (clean) training data."""
        X_scaled = self.scaler.fit_transform(X)
        behavior_features = X_scaled[:, self._behavior_feature_indices]
        self.behavior_detector.fit(behavior_features)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> List[dict]:
        """Predict three-way labels for each sample."""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        results = []

        for i in range(len(X)):
            result = self._predict_single(X[i], X_scaled[i])
            results.append(result)

        return results

    def _predict_single(self, x_raw: np.ndarray, x_scaled: np.ndarray) -> dict:
        """Three-way decision for a single window."""
        # Step 1: Check integrity signals
        integrity_score = self._compute_integrity_score(x_raw)

        # Step 2: Check behavioral anomaly (only meaningful if integrity is OK)
        behavior_features = x_scaled[self._behavior_feature_indices].reshape(1, -1)
        if_score = self.behavior_detector.decision_function(behavior_features)[0]
        if_pred = self.behavior_detector.predict(behavior_features)[0]

        # Normalize IF score to [0, 1] range (more negative = more anomalous)
        behavior_score = max(0.0, min(1.0, 0.5 - if_score))

        # Step 3: Three-way decision
        needs_audit = x_raw[18] > self.audit_threshold  # sketch_residual

        if integrity_score > self.integrity_threshold:
            label = self.INTEGRITY_FAILURE
        elif if_pred == -1:
            label = self.BEHAVIORAL_ANOMALY
        else:
            label = self.NORMAL

        return {
            "label": label,
            "integrity_score": integrity_score,
            "behavior_score": behavior_score,
            "sketch_residual": float(x_raw[18]),
            "needs_audit": needs_audit,
        }

    def _compute_integrity_score(self, x_raw: np.ndarray) -> float:
        """Compute integrity anomaly score from provenance signals."""
        sketch_residual = x_raw[18]
        cell_coverage = x_raw[19]
        flags = x_raw[20:24]

        # Any hard flag immediately triggers integrity failure
        if np.any(flags > 0.5):
            return 0.9

        # Weighted combination for soft signals
        score = (
            0.45 * min(sketch_residual / 0.3, 1.0) +
            0.35 * (1.0 - cell_coverage) +
            0.20 * np.mean(flags)
        )
        return float(np.clip(score, 0.0, 1.0))

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return numeric labels: 0=normal, 1=integrity_failure, 2=behavioral_anomaly."""
        results = self.predict(X)
        label_map = {self.NORMAL: 0, self.INTEGRITY_FAILURE: 1, self.BEHAVIORAL_ANOMALY: 2}
        return np.array([label_map[r["label"]] for r in results])


class BaselineDetector:
    """Baseline: standard Isolation Forest without integrity awareness."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100,
                 random_state: int = 42):
        self.detector = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        self.detector.fit(X_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        preds = self.detector.predict(X_s)
        return np.where(preds == -1, 1, 0)  # 1=anomaly, 0=normal

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.detector.decision_function(X_s)


class BaselineLOF:
    """Baseline: Local Outlier Factor."""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.detector = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        self.detector.fit(X_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        preds = self.detector.predict(X_s)
        return np.where(preds == -1, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.detector.decision_function(X_s)


class BaselineOCSVM:
    """Baseline: One-Class SVM on trajectory features."""

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str = "scale"):
        self.detector = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        self.detector.fit(X_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        preds = self.detector.predict(X_s)
        return np.where(preds == -1, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.detector.decision_function(X_s)


class BaselineTRAOD:
    """Baseline: TRAOD-style partition-and-detect trajectory outlier detection.

    Adapted to work on feature vectors: partitions the feature space into
    segments and uses distance-based outlier detection with perpendicular
    and angular distance metrics on trajectory feature sequences.

    Reference: Lee, Han, Li. "Trajectory Outlier Detection: A Partition-and-Detect
    Framework." ICDE 2008.
    """

    def __init__(self, distance_threshold: float = None, fraction_threshold: float = 0.3,
                 min_neighbors: int = 5):
        self.distance_threshold = distance_threshold
        self.fraction_threshold = fraction_threshold
        self.min_neighbors = min_neighbors
        self.scaler = StandardScaler()
        self._train_data = None

    def fit(self, X: np.ndarray):
        self._train_data = self.scaler.fit_transform(X)
        if self.distance_threshold is None:
            # Auto-calibrate: use mean + 2*std of pairwise distances
            dists = self._pairwise_distances(self._train_data)
            self.distance_threshold = np.mean(dists) + 2.0 * np.std(dists)
        return self

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances (sample for efficiency)."""
        n = len(X)
        if n > 500:
            idx = np.random.choice(n, 500, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X_sample, metric="euclidean")
        return D[np.triu_indices_from(D, k=1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        preds = np.zeros(len(X_s), dtype=int)
        for i in range(len(X_s)):
            dists = np.linalg.norm(self._train_data - X_s[i], axis=1)
            n_close = np.sum(dists < self.distance_threshold)
            if n_close < self.min_neighbors:
                preds[i] = 1  # anomaly
        return preds

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: negative = more anomalous (consistent with sklearn convention)."""
        X_s = self.scaler.transform(X)
        scores = np.zeros(len(X_s))
        for i in range(len(X_s)):
            dists = np.linalg.norm(self._train_data - X_s[i], axis=1)
            dists_sorted = np.sort(dists)
            # Average distance to k nearest neighbors (lower = more normal)
            k = min(self.min_neighbors, len(dists_sorted))
            avg_dist = np.mean(dists_sorted[:k])
            scores[i] = -avg_dist  # negative = more anomalous
        return scores


class BaselineIBAT:
    """Baseline: iBAT-style isolation-based anomalous trajectory detection.

    Adapted to work on feature vectors: discretizes feature space into cells
    and computes isolation score based on cell occupancy.

    Reference: Chen et al. "iBOAT: Isolation-Based Online Anomalous Trajectory
    Detection." IEEE TITS, 2013.
    """

    def __init__(self, n_bins: int = 10, anomaly_threshold: float = None,
                 min_support: int = 3):
        self.n_bins = n_bins
        self.anomaly_threshold = anomaly_threshold
        self.min_support = min_support
        self.scaler = StandardScaler()
        self._cell_counts = None
        self._bin_edges = None

    def fit(self, X: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        n_features = X_s.shape[1]
        self._bin_edges = []
        self._cell_counts = {}

        # Create bin edges for each feature
        for j in range(n_features):
            edges = np.linspace(X_s[:, j].min() - 1e-10, X_s[:, j].max() + 1e-10,
                                self.n_bins + 1)
            self._bin_edges.append(edges)

        # Count cell occupancy for each feature dimension
        for j in range(n_features):
            bins = np.digitize(X_s[:, j], self._bin_edges[j]) - 1
            bins = np.clip(bins, 0, self.n_bins - 1)
            counts = np.bincount(bins, minlength=self.n_bins)
            self._cell_counts[j] = counts

        # Auto-calibrate threshold
        if self.anomaly_threshold is None:
            train_scores = self._compute_scores(X_s)
            self.anomaly_threshold = np.percentile(train_scores, 5)

        return self

    def _compute_scores(self, X_s: np.ndarray) -> np.ndarray:
        """Compute isolation scores (higher = more normal)."""
        n_features = X_s.shape[1]
        scores = np.zeros(len(X_s))
        for i in range(len(X_s)):
            support_sum = 0
            for j in range(n_features):
                bin_idx = np.digitize(X_s[i, j], self._bin_edges[j]) - 1
                bin_idx = max(0, min(bin_idx, self.n_bins - 1))
                support_sum += self._cell_counts[j][bin_idx]
            scores[i] = support_sum / n_features
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        scores = self._compute_scores(X_s)
        return np.where(scores < self.anomaly_threshold, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: negative of support (more negative = more anomalous)."""
        X_s = self.scaler.transform(X)
        scores = self._compute_scores(X_s)
        return scores - self.anomaly_threshold  # negative = anomalous
