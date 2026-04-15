"""Quick GeoLife experiment with multi-modal trajectory data."""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from windowproof.capture.secure_capture import SecureCaptureModule
from windowproof.capture.checkpoint_extractor import compute_checkpoint_features
from windowproof.blockchain.smart_contract import WindowChainContract
from windowproof.blockchain.window_chain import WindowChain
from windowproof.detection.sketch_consistency import (
    reconstruct_sketch_from_checkpoints, compute_sketch_residual, build_detection_features,
)
from windowproof.detection.three_way_detector import (
    ThreeWayDetector, BaselineDetector, BaselineLOF,
    BaselineOCSVM, BaselineTRAOD, BaselineIBAT,
)
from windowproof.attacks.attack_simulator import AttackSimulator
from windowproof.utils.real_data_loader import load_preprocessed_csv
from windowproof.utils.metrics import (
    three_way_metrics, binary_anomaly_metrics, attack_coverage_report,
    blockchain_metrics_summary,
)


def process_traj(traj, capture, chain, dev_id):
    windows = capture.process_trajectory(traj)
    records = []
    for w in windows:
        cr = chain.commit_window(dev_id, w["window_id"], w["raw_points"],
                                  {k: v for k, v in w["sketch"].items() if k != "geohash_set"})
        cp_pts = [w["raw_points"][i] for i in w["checkpoint_indices"] if i < len(w["raw_points"])]
        cp_f = compute_checkpoint_features(w["raw_points"], w["checkpoint_indices"])
        recon = reconstruct_sketch_from_checkpoints(cp_pts)
        resid = compute_sketch_residual(w["sketch"], recon)
        flags = {"window_missing": not cr.get("success", True), "late_commitment": False,
                 "proof_failed": False, "density_violation": cp_f["checkpoint_density"] < 0.1}
        feat = build_detection_features(cp_f, w["sketch"], resid, flags)
        records.append({"features": feat, "sketch": w["sketch"]})
    return records


def main():
    seed = 42
    np.random.seed(seed)

    print("Loading GeoLife data...")
    trajs = load_preprocessed_csv("data/geolife_sample.csv", max_trajectories=300,
                                   min_length=15, seed=seed)
    print(f"Loaded {len(trajs)} trajectories")

    n_test = int(len(trajs) * 0.3)
    train_trajs = trajs[:-n_test]
    test_trajs = trajs[-n_test:]

    print(f"Train: {len(train_trajs)}, Test: {len(test_trajs)}")

    # Process training data
    print("Processing training data...")
    contract = WindowChainContract(min_checkpoint_density=0.1)
    chain = WindowChain(contract)
    capture = SecureCaptureModule(device_id="train", window_size_sec=300)

    train_features = []
    max_windows_per_traj = 5
    for t in train_trajs:
        recs = process_traj(t, capture, chain, "train")
        for r in recs[:max_windows_per_traj]:
            train_features.append(r["features"])
    X_train = np.array(train_features)
    print(f"Training windows: {len(X_train)}")

    # Generate test data with attacks
    print("Generating attacked test data...")
    attacker = AttackSimulator(seed=seed)
    behavioral_attacks = ["detour", "loop", "abnormal_stop", "speed_burst", "teleport"]
    integrity_attacks = ["point_deletion", "point_injection", "timestamp_shift", "replay"]

    test_features, test_labels, test_attack_types = [], [], []

    for i, traj in enumerate(test_trajs):
        if np.random.random() < 0.3:
            if np.random.random() < 0.5:
                atype = np.random.choice(behavioral_attacks)
                attacked, info = attacker.generate_attack(traj, atype)
                label = 2
            else:
                atype = np.random.choice(integrity_attacks)
                attacked, info = attacker.generate_attack(traj, atype)
                label = 1

            cap_orig = SecureCaptureModule(device_id=f"o{i}", window_size_sec=300)
            cap_atk = SecureCaptureModule(device_id=f"a{i}", window_size_sec=300)
            orig_wins = cap_orig.process_trajectory(traj)
            atk_wins = cap_atk.process_trajectory(attacked)

            for j in range(min(len(orig_wins), len(atk_wins), max_windows_per_traj)):
                ow, aw = orig_wins[j], atk_wins[j]
                sketch = ow["sketch"] if label == 1 else aw["sketch"]
                cp_pts = [aw["raw_points"][k] for k in aw["checkpoint_indices"]
                          if k < len(aw["raw_points"])]
                cp_f = compute_checkpoint_features(aw["raw_points"], aw["checkpoint_indices"])
                recon = reconstruct_sketch_from_checkpoints(cp_pts)
                resid = compute_sketch_residual(sketch, recon)
                flags = {"window_missing": False, "late_commitment": False,
                         "proof_failed": label == 1, "density_violation": False}
                feat = build_detection_features(cp_f, sketch, resid, flags)
                test_features.append(feat)
                test_labels.append(label)
                test_attack_types.append(atype)
        else:
            cap2 = SecureCaptureModule(device_id=f"n{i}", window_size_sec=300)
            cont2 = WindowChainContract(min_checkpoint_density=0.1)
            ch2 = WindowChain(cont2)
            recs = process_traj(traj, cap2, ch2, f"n{i}")
            for r in recs[:max_windows_per_traj]:
                test_features.append(r["features"])
                test_labels.append(0)
                test_attack_types.append("none")

    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    print(f"Test: {len(X_test)} windows (Normal: {np.sum(y_test==0)}, "
          f"Integrity: {np.sum(y_test==1)}, Behavioral: {np.sum(y_test==2)})")

    # WindowProof
    print("\n--- WindowProof ---")
    det = ThreeWayDetector(n_estimators=100, behavior_contamination=0.05, random_state=seed)
    det.fit(X_train)
    t0 = time.perf_counter()
    y_wp = det.predict_labels(X_test)
    wp_time = (time.perf_counter() - t0) * 1000
    wp_metrics = three_way_metrics(y_test, y_wp)
    print(f"Three-way macro-F1: {wp_metrics['macro_f1']:.4f}")
    print(f"Inference time: {wp_time:.1f} ms")

    # Baselines
    y_bin = (y_test > 0).astype(int)

    print("\n--- Baseline IF (Full) ---")
    bif = BaselineDetector(contamination=0.05, random_state=seed)
    bif.fit(X_train)
    y_bif = bif.predict(X_test)
    bif_s = -bif.decision_function(X_test)
    bif_m = binary_anomaly_metrics(y_bin, bif_s, y_bif)
    print(f"F1: {bif_m['f1']:.4f}, AUROC: {bif_m['auroc']:.4f}")

    print("\n--- Baseline LOF ---")
    blof = BaselineLOF(n_neighbors=20, contamination=0.05)
    blof.fit(X_train)
    y_blof = blof.predict(X_test)
    blof_s = -blof.decision_function(X_test)
    blof_m = binary_anomaly_metrics(y_bin, blof_s, y_blof)
    print(f"F1: {blof_m['f1']:.4f}, AUROC: {blof_m['auroc']:.4f}")

    print("\n--- Baseline OC-SVM ---")
    bocsvm = BaselineOCSVM(nu=0.05)
    bocsvm.fit(X_train)
    y_bocsvm = bocsvm.predict(X_test)
    bocsvm_s = -bocsvm.decision_function(X_test)
    bocsvm_m = binary_anomaly_metrics(y_bin, bocsvm_s, y_bocsvm)
    print(f"F1: {bocsvm_m['f1']:.4f}, AUROC: {bocsvm_m['auroc']:.4f}")

    print("\n--- Baseline TRAOD ---")
    btraod = BaselineTRAOD(min_neighbors=5)
    btraod.fit(X_train)
    y_btraod = btraod.predict(X_test)
    btraod_s = -btraod.decision_function(X_test)
    btraod_m = binary_anomaly_metrics(y_bin, btraod_s, y_btraod)
    print(f"F1: {btraod_m['f1']:.4f}, AUROC: {btraod_m['auroc']:.4f}")

    print("\n--- Baseline iBAT ---")
    bibat = BaselineIBAT(n_bins=10, min_support=3)
    bibat.fit(X_train)
    y_bibat = bibat.predict(X_test)
    bibat_s = -bibat.decision_function(X_test)
    bibat_m = binary_anomaly_metrics(y_bin, bibat_s, y_bibat)
    print(f"F1: {bibat_m['f1']:.4f}, AUROC: {bibat_m['auroc']:.4f}")

    print("\n--- Baseline IF (Sparse Only) ---")
    bsp = BaselineDetector(contamination=0.05, random_state=seed)
    bsp.fit(X_train[:, :9])
    y_bsp = bsp.predict(X_test[:, :9])
    bsp_s = -bsp.decision_function(X_test[:, :9])
    bsp_m = binary_anomaly_metrics(y_bin, bsp_s, y_bsp)
    print(f"F1: {bsp_m['f1']:.4f}, AUROC: {bsp_m['auroc']:.4f}")

    print("\n--- Baseline IF (Sketch Only) ---")
    bsk = BaselineDetector(contamination=0.05, random_state=seed)
    bsk.fit(X_train[:, 9:18])
    y_bsk = bsk.predict(X_test[:, 9:18])
    bsk_s = -bsk.decision_function(X_test[:, 9:18])
    bsk_m = binary_anomaly_metrics(y_bin, bsk_s, y_bsk)
    print(f"F1: {bsk_m['f1']:.4f}, AUROC: {bsk_m['auroc']:.4f}")

    # Attack coverage
    print("\n--- Attack Coverage ---")
    coverage = attack_coverage_report(test_attack_types, y_test, y_wp)
    for atype, stats in sorted(coverage.items()):
        if atype != "none":
            print(f"  {atype}: {stats['detection_rate']:.1%} ({stats['detected']}/{stats['total']})")

    # Blockchain metrics
    bc_m = blockchain_metrics_summary(contract.get_metrics())
    print(f"\nBlockchain: {bc_m['storage_per_commitment_bytes']:.0f} bytes/commit, "
          f"gas: {bc_m['total_gas_cost']:.0f}")

    # Save results
    os.makedirs("results_geolife", exist_ok=True)
    results = {
        "dataset": "GeoLife (Multi-modal GPS)",
        "trajectories": len(trajs),
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "windowproof": wp_metrics,
        "baseline_if": bif_m,
        "baseline_lof": blof_m,
        "baseline_ocsvm": bocsvm_m,
        "baseline_traod": btraod_m,
        "baseline_ibat": bibat_m,
        "baseline_sparse_if": bsp_m,
        "baseline_sketch_if": bsk_m,
        "attack_coverage": coverage,
        "blockchain_metrics": bc_m,
    }
    with open("results_geolife/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to results_geolife/results.json")


if __name__ == "__main__":
    main()
