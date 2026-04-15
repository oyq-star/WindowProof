"""Merkle tree implementation for trajectory segment integrity verification."""

import hashlib
from typing import List, Optional


class MerkleTree:
    """Binary Merkle tree for verifying trajectory data integrity."""

    def __init__(self, data_blocks: List[bytes], hash_algo: str = "sha256"):
        self.hash_algo = hash_algo
        self.leaves = [self._hash(block) for block in data_blocks]
        self.tree = self._build_tree(self.leaves)
        self.root = self.tree[-1][0] if self.tree else b""

    def _hash(self, data: bytes) -> bytes:
        h = hashlib.new(self.hash_algo)
        h.update(data)
        return h.digest()

    def _build_tree(self, leaves: List[bytes]) -> List[List[bytes]]:
        if not leaves:
            return []
        tree = [leaves]
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash(left + right))
            tree.append(next_level)
            current_level = next_level
        return tree

    def get_root(self) -> bytes:
        return self.root

    def get_proof(self, index: int) -> List[tuple]:
        """Get Merkle proof for leaf at given index.
        Returns list of (sibling_hash, direction) tuples."""
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range")
        proof = []
        idx = index
        for level in self.tree[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1
                direction = "right"
            else:
                sibling_idx = idx - 1
                direction = "left"
            if sibling_idx < len(level):
                proof.append((level[sibling_idx], direction))
            else:
                proof.append((level[idx], "right"))
            idx = idx // 2
        return proof

    @staticmethod
    def verify_proof(leaf_hash: bytes, proof: List[tuple],
                     root: bytes, hash_algo: str = "sha256") -> bool:
        """Verify a Merkle proof against the root hash."""
        h = hashlib.new(hash_algo)
        current = leaf_hash
        for sibling, direction in proof:
            h = hashlib.new(hash_algo)
            if direction == "left":
                h.update(sibling + current)
            else:
                h.update(current + sibling)
            current = h.digest()
        return current == root


def hash_gps_point(lat: float, lon: float, timestamp: float) -> bytes:
    """Hash a single GPS point for Merkle tree leaf."""
    data = f"{lat:.8f},{lon:.8f},{timestamp:.3f}".encode("utf-8")
    return data


def hash_trajectory_window(points: List[tuple]) -> bytes:
    """Hash a window of GPS points: [(lat, lon, timestamp), ...]."""
    data = "|".join(f"{p[0]:.8f},{p[1]:.8f},{p[2]:.3f}" for p in points)
    return data.encode("utf-8")
