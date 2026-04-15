"""WindowChain: blockchain commitment and verification pipeline."""

import hashlib
import hmac
from typing import List, Optional

from .merkle_tree import MerkleTree, hash_trajectory_window
from .smart_contract import WindowChainContract


class WindowChain:
    """Manages the full commitment lifecycle for trajectory windows."""

    def __init__(self, contract: WindowChainContract, device_key: bytes = b"default_key"):
        self.contract = contract
        self.device_key = device_key
        self.prev_window_hashes = {}

    def _sign(self, data: bytes) -> bytes:
        return hmac.new(self.device_key, data, hashlib.sha256).digest()

    def commit_window(self, device_id: str, window_id: int,
                      raw_points: List[tuple], sketch_data: dict) -> dict:
        """Commit a raw window: build Merkle tree, compute hashes, anchor on chain."""
        data_blocks = [
            f"{p[0]:.8f},{p[1]:.8f},{p[2]:.3f}".encode("utf-8")
            for p in raw_points
        ]
        merkle = MerkleTree(data_blocks)
        merkle_root = merkle.get_root()

        sketch_str = str(sorted(sketch_data.items()))
        sketch_hash = hashlib.sha256(sketch_str.encode("utf-8")).digest()

        prev_hash = self.prev_window_hashes.get(device_id, b"\x00" * 32)

        payload = merkle_root + sketch_hash + prev_hash + f"{window_id}".encode()
        signature = self._sign(payload)

        result = self.contract.commit_window(
            device_id=device_id,
            window_id=window_id,
            merkle_root=merkle_root,
            sketch_hash=sketch_hash,
            sketch_data=sketch_data,
            prev_window_hash=prev_hash,
            signature=signature,
        )

        if result["success"]:
            window_hash = hashlib.sha256(merkle_root + sketch_hash).digest()
            self.prev_window_hashes[device_id] = window_hash

        return {**result, "merkle_tree": merkle, "merkle_root": merkle_root}

    def disclose_checkpoints(self, device_id: str, window_id: int,
                             checkpoint_indices: List[int],
                             raw_points: List[tuple],
                             merkle_tree: MerkleTree,
                             total_points: int) -> dict:
        """Submit sparse checkpoint disclosure with Merkle proofs."""
        checkpoints = []
        proofs = []
        for idx in checkpoint_indices:
            if idx < len(raw_points):
                p = raw_points[idx]
                checkpoints.append({"lat": p[0], "lon": p[1], "timestamp": p[2], "index": idx})
                proofs.append(merkle_tree.get_proof(idx))

        return self.contract.submit_sparse_disclosure(
            device_id=device_id,
            window_id=window_id,
            checkpoints=checkpoints,
            checkpoint_proofs=proofs,
            total_points=total_points,
        )

    def verify_checkpoint(self, checkpoint: dict, proof: list,
                          merkle_root: bytes) -> bool:
        """Verify a single checkpoint against its Merkle proof."""
        data = f"{checkpoint['lat']:.8f},{checkpoint['lon']:.8f},{checkpoint['timestamp']:.3f}"
        leaf_hash = hashlib.sha256(data.encode("utf-8")).digest()
        return MerkleTree.verify_proof(leaf_hash, proof, merkle_root)
