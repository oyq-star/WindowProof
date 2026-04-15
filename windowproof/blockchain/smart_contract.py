"""Smart contract simulation for WindowChain commitment protocol."""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WindowCommitment:
    device_id: str
    window_id: int
    merkle_root: bytes
    sketch_hash: bytes
    sketch_data: Optional[dict]
    prev_window_hash: bytes
    timestamp: float
    signature: bytes
    status: str = "committed"  # committed, sparse_disclosed, audited, failed


@dataclass
class SparseDisclosure:
    device_id: str
    window_id: int
    checkpoints: List[dict]
    checkpoint_proofs: List[list]
    checkpoint_count: int


@dataclass
class BlockchainMetrics:
    total_commitments: int = 0
    total_disclosures: int = 0
    total_audits: int = 0
    total_failures: int = 0
    total_storage_bytes: int = 0
    total_verification_time_ms: float = 0.0
    gas_per_commit: float = 45000.0
    gas_per_disclose: float = 30000.0
    gas_per_audit: float = 60000.0


class WindowChainContract:
    """Simulated smart contract for WindowProof protocol."""

    def __init__(self, commitment_deadline_sec: float = 600,
                 max_window_gap: int = 1,
                 min_checkpoint_density: float = 0.1):
        self.commitment_deadline = commitment_deadline_sec
        self.max_window_gap = max_window_gap
        self.min_checkpoint_density = min_checkpoint_density
        self.commitments: Dict[str, Dict[int, WindowCommitment]] = {}
        self.disclosures: Dict[str, Dict[int, SparseDisclosure]] = {}
        self.metrics = BlockchainMetrics()

    def commit_window(self, device_id: str, window_id: int,
                      merkle_root: bytes, sketch_hash: bytes,
                      sketch_data: Optional[dict],
                      prev_window_hash: bytes,
                      signature: bytes) -> dict:
        """Anchor a window commitment on chain."""
        start_time = time.perf_counter()

        if device_id not in self.commitments:
            self.commitments[device_id] = {}

        # Check contiguous window IDs
        if self.commitments[device_id]:
            last_id = max(self.commitments[device_id].keys())
            gap = window_id - last_id
            if gap > self.max_window_gap + 1:
                self.metrics.total_failures += 1
                return {
                    "success": False,
                    "reason": f"Window gap {gap} exceeds max {self.max_window_gap}",
                    "type": "missing_window"
                }

        commitment = WindowCommitment(
            device_id=device_id,
            window_id=window_id,
            merkle_root=merkle_root,
            sketch_hash=sketch_hash,
            sketch_data=sketch_data,
            prev_window_hash=prev_window_hash,
            timestamp=time.time(),
            signature=signature,
        )
        self.commitments[device_id][window_id] = commitment
        self.metrics.total_commitments += 1

        storage = len(merkle_root) + len(sketch_hash) + len(prev_window_hash) + len(signature) + 64
        if sketch_data:
            storage += len(str(sketch_data))
        self.metrics.total_storage_bytes += storage

        elapsed = (time.perf_counter() - start_time) * 1000
        self.metrics.total_verification_time_ms += elapsed

        return {"success": True, "window_id": window_id, "storage_bytes": storage}

    def submit_sparse_disclosure(self, device_id: str, window_id: int,
                                 checkpoints: List[dict],
                                 checkpoint_proofs: List[list],
                                 total_points: int) -> dict:
        """Submit sparse checkpoint disclosure for a committed window."""
        start_time = time.perf_counter()

        if device_id not in self.commitments or window_id not in self.commitments[device_id]:
            return {"success": False, "reason": "No commitment found"}

        density = len(checkpoints) / max(total_points, 1)
        if density < self.min_checkpoint_density:
            self.metrics.total_failures += 1
            return {
                "success": False,
                "reason": f"Checkpoint density {density:.3f} below minimum {self.min_checkpoint_density}",
                "type": "insufficient_density"
            }

        disclosure = SparseDisclosure(
            device_id=device_id,
            window_id=window_id,
            checkpoints=checkpoints,
            checkpoint_proofs=checkpoint_proofs,
            checkpoint_count=len(checkpoints),
        )

        if device_id not in self.disclosures:
            self.disclosures[device_id] = {}
        self.disclosures[device_id][window_id] = disclosure
        self.commitments[device_id][window_id].status = "sparse_disclosed"
        self.metrics.total_disclosures += 1

        elapsed = (time.perf_counter() - start_time) * 1000
        self.metrics.total_verification_time_ms += elapsed

        return {"success": True, "density": density}

    def request_audit(self, device_id: str, window_id: int) -> dict:
        """Request full raw data audit for a suspicious window."""
        if device_id not in self.commitments or window_id not in self.commitments[device_id]:
            return {"success": False, "reason": "No commitment found"}

        self.commitments[device_id][window_id].status = "audited"
        self.metrics.total_audits += 1
        return {"success": True, "status": "audit_requested"}

    def check_continuity(self, device_id: str) -> List[dict]:
        """Check for missing or late windows."""
        if device_id not in self.commitments:
            return []

        windows = sorted(self.commitments[device_id].keys())
        gaps = []
        for i in range(1, len(windows)):
            gap = windows[i] - windows[i - 1]
            if gap > 1:
                for missing_id in range(windows[i - 1] + 1, windows[i]):
                    gaps.append({
                        "device_id": device_id,
                        "missing_window_id": missing_id,
                        "type": "missing_window"
                    })
        return gaps

    def get_metrics(self) -> dict:
        m = self.metrics
        return {
            "total_commitments": m.total_commitments,
            "total_disclosures": m.total_disclosures,
            "total_audits": m.total_audits,
            "total_failures": m.total_failures,
            "total_storage_bytes": m.total_storage_bytes,
            "avg_verification_time_ms": (
                m.total_verification_time_ms / max(m.total_commitments + m.total_disclosures, 1)
            ),
            "estimated_gas_cost": (
                m.total_commitments * m.gas_per_commit +
                m.total_disclosures * m.gas_per_disclose +
                m.total_audits * m.gas_per_audit
            ),
        }
