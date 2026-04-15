"""Main experiment: WindowProof vs baselines on synthetic + semi-synthetic data."""

import os
import sys
import json
import time
import argparse
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from windowproof.capture.secure_capture import SecureCaptureModule
from windowproof.capture.window_sketch import compute_window_sketch, sketch_to_vector
from windowproof.capture.checkpoint_extractor import extract_checkpoints, compute_checkpoint_features
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
from windowproof.utils.data_loader import generate_synthetic_trajectories
from windowproof.utils.real_data_loader import load_porto_raw, load_tdrive_raw, load_preprocessed_csv
from windowproof.utils.metrics import (
    three_way_metrics, binary_anomaly_metrics,
    attack_coverage_report, blockchain_metrics_summary,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_trajectory_through_pipeline(trajectory, capture_module, window_chain, device_id):
    """Process a single trajectory through the full WindowProof pipeline."""
    windows = capture_module.process_trajectory(trajectory)
    window_records = []

    for w in windows:
        commit_result = window_chain.commit_window(
            device_id=device_id,
            window_id=w["window_id"],
            raw_points=w["raw_points"],
            sketch_data={k: v for k, v in w["sketch"].items() if k != "geohash_set"},
        )

        checkpoint_pts = [w["raw_points"][i] for i in w["checkpoint_indices"]
                          if i < len(w["raw_points"])]
        cp_features = compute_checkpoint_features(w["raw_points"], w["checkpoint_indices"])

        reconstructed = reconstruct_sketch_from_checkpoints(checkpoint_pts)
        residual = compute_sketch_residual(w["sketch"], reconstructed)

        integrity_flags = {
            "window_missing": not commit_result.get("success", True),
            "late_commitment": False,
            "proof_failed": False,
            "density_violation": cp_features["checkpoint_density"] < 0.1,
        }

        features = build_detection_features(
            cp_features, w["sketch"], residual, integrity_flags
        )

        window_records.append({
            "window_id": w["window_id"],
            "features": features,
            "sketch": w["sketch"],
            "residual": residual,
            "cp_features": cp_features,
        })

    return window_records


def run_experiment(config: dict, output_dir: str, data_source: str = "synthetic"):
    """Run the main experiment."""
    seed = config["experiment"]["random_seed"]
    np.random.seed(seed)

    print("=" * 60)
    print("WindowProof Experiment")
    print("=" * 60)

    # Step 1: Load or generate data
    print("\n[1/6] Generating/loading trajectory data...")
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if data_source == "synthetic":
        trajectories = generate_synthetic_trajectories(
            n_trajectories=1000, min_length=20, max_length=100, seed=seed
        )
    elif data_source == "porto":
        porto_csv = data_dir / "porto" / "train.csv"
        if not porto_csv.exists():
            print(f"  Porto data not found at {porto_csv}. Falling back to synthetic.")
            trajectories = generate_synthetic_trajectories(
                n_trajectories=1000, min_length=20, max_length=100, seed=seed
            )
        else:
            trajectories = load_porto_raw(str(porto_csv), max_trajectories=2000,
                                          min_length=15, seed=seed)
            print(f"  Loaded {len(trajectories)} Porto trajectories")
    elif data_source == "tdrive":
        tdrive_dir = data_dir / "tdrive"
        tdrive_csv = data_dir / "tdrive_sample.csv"
        if tdrive_csv.exists():
            trajectories = load_preprocessed_csv(str(tdrive_csv), max_trajectories=2000,
                                                  min_length=15, seed=seed)
        elif tdrive_dir.exists():
            trajectories = load_tdrive_raw(str(tdrive_dir), max_trajectories=2000,
                                           min_length=15, seed=seed)
        else:
            print(f"  T-Drive data not found. Falling back to synthetic.")
            trajectories = generate_synthetic_trajectories(
                n_trajectories=1000, min_length=20, max_length=100, seed=seed
            )
        print(f"  Loaded {len(trajectories)} T-Drive trajectories")
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    n_total = len(trajectories)
    n_test = int(n_total * config["experiment"]["test_ratio"])
    n_train = n_total - n_test
    train_trajs = trajectories[:n_train]
    test_trajs = trajectories[n_train:]
    print(f"  Total: {n_total}, Train: {n_train}, Test: {n_test}")

    # Step 2: Process clean training data through pipeline
    print("\n[2/6] Processing clean training data through WindowProof pipeline...")
    contract = WindowChainContract(
        min_checkpoint_density=config["checkpoint"]["min_checkpoint_density"],
    )
    chain = WindowChain(contract)
    capture = SecureCaptureModule(
        device_id="train_device",
        window_size_sec=config["data"]["window_size_sec"],
    )

    train_features = []
    for traj in train_trajs:
        records = process_trajectory_through_pipeline(traj, capture, chain, "train_device")
        for r in records:
            train_features.append(r["features"])

    X_train = np.array(train_features)
    print(f"  Training windows: {len(X_train)}, features: {X_train.shape[1]}")

    # Step 3: Generate test data with attacks
    print("\n[3/6] Generating test data with attacks...")
    attacker = AttackSimulator(seed=seed)
    anomaly_ratio = config["experiment"]["anomaly_ratio"]

    test_features = []
    test_labels = []  # 0=normal, 1=integrity, 2=behavioral
    test_attack_types = []

    behavioral_attacks = ["detour", "loop", "abnormal_stop", "speed_burst", "teleport"]
    integrity_attacks = ["point_deletion", "point_injection", "timestamp_shift", "replay"]

    capture_test = SecureCaptureModule(
        device_id="test_device",
        window_size_sec=config["data"]["window_size_sec"],
    )
    contract_test = WindowChainContract(
        min_checkpoint_density=config["checkpoint"]["min_checkpoint_density"],
    )
    chain_test = WindowChain(contract_test)

    for i, traj in enumerate(test_trajs):
        is_anomaly = np.random.random() < anomaly_ratio

        if is_anomaly:
            # 50% behavioral, 50% integrity
            if np.random.random() < 0.5:
                attack_type = np.random.choice(behavioral_attacks)
                attacked_traj, info = attacker.generate_attack(traj, attack_type)
                label = 2
            else:
                attack_type = np.random.choice(integrity_attacks)
                attacked_traj, info = attacker.generate_attack(traj, attack_type)
                label = 1

            # For integrity attacks, the committed sketch is from ORIGINAL data
            # but checkpoints are from ATTACKED data (simulating post-capture tampering)
            original_windows = capture_test.process_trajectory(traj)
            attacked_capture = SecureCaptureModule(
                device_id=f"test_{i}", window_size_sec=config["data"]["window_size_sec"]
            )
            attacked_windows = attacked_capture.process_trajectory(attacked_traj)

            for j in range(min(len(original_windows), len(attacked_windows))):
                orig_w = original_windows[j]
                atk_w = attacked_windows[j]

                # For integrity attacks: committed sketch is from original, checkpoints from attacked
                if label == 1:
                    committed_sketch = orig_w["sketch"]
                    checkpoint_pts = [atk_w["raw_points"][k] for k in atk_w["checkpoint_indices"]
                                      if k < len(atk_w["raw_points"])]
                else:
                    committed_sketch = atk_w["sketch"]
                    checkpoint_pts = [atk_w["raw_points"][k] for k in atk_w["checkpoint_indices"]
                                      if k < len(atk_w["raw_points"])]

                cp_features = compute_checkpoint_features(
                    atk_w["raw_points"], atk_w["checkpoint_indices"]
                )
                reconstructed = reconstruct_sketch_from_checkpoints(checkpoint_pts)
                residual = compute_sketch_residual(committed_sketch, reconstructed)

                integrity_flags = {
                    "window_missing": False,
                    "late_commitment": False,
                    "proof_failed": label == 1,
                    "density_violation": cp_features["checkpoint_density"] < 0.1,
                }

                features = build_detection_features(
                    cp_features, committed_sketch, residual, integrity_flags
                )

                test_features.append(features)
                test_labels.append(label)
                test_attack_types.append(attack_type)
        else:
            records = process_trajectory_through_pipeline(
                traj, capture_test, chain_test, f"test_{i}"
            )
            for r in records:
                test_features.append(r["features"])
                test_labels.append(0)
                test_attack_types.append("none")

    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    print(f"  Test windows: {len(X_test)}")
    print(f"  Normal: {np.sum(y_test == 0)}, Integrity: {np.sum(y_test == 1)}, "
          f"Behavioral: {np.sum(y_test == 2)}")

    # Step 4: Train and evaluate WindowProof
    print("\n[4/6] Training and evaluating WindowProof detector...")
    detector = ThreeWayDetector(
        n_estimators=config["detection"]["isolation_forest"]["n_estimators"],
        behavior_contamination=config["detection"]["isolation_forest"]["contamination"],
        audit_threshold=config["detection"]["audit_threshold"],
        random_state=seed,
    )
    detector.fit(X_train)

    t_start = time.perf_counter()
    y_pred_wp = detector.predict_labels(X_test)
    wp_inference_time = (time.perf_counter() - t_start) * 1000

    wp_results = three_way_metrics(y_test, y_pred_wp)
    wp_results["inference_time_ms"] = wp_inference_time
    print(f"  WindowProof three-way macro-F1: {wp_results['macro_f1']:.4f}")

    # Step 5: Train and evaluate baselines
    print("\n[5/6] Evaluating baselines...")
    results_all = {"windowproof": wp_results}

    # Baseline 1: Standard IF on all features (no integrity awareness)
    baseline_if = BaselineDetector(
        contamination=config["detection"]["isolation_forest"]["contamination"],
        n_estimators=config["detection"]["isolation_forest"]["n_estimators"],
        random_state=seed,
    )
    baseline_if.fit(X_train)
    y_pred_if = baseline_if.predict(X_test)
    y_binary_true = (y_test > 0).astype(int)
    if_scores = -baseline_if.decision_function(X_test)
    results_all["baseline_if"] = binary_anomaly_metrics(y_binary_true, if_scores, y_pred_if)
    print(f"  Baseline IF F1: {results_all['baseline_if']['f1']:.4f}")

    # Baseline 2: LOF
    baseline_lof = BaselineLOF(
        n_neighbors=config["detection"]["lof"]["n_neighbors"],
        contamination=config["detection"]["lof"]["contamination"],
    )
    baseline_lof.fit(X_train)
    y_pred_lof = baseline_lof.predict(X_test)
    lof_scores = -baseline_lof.decision_function(X_test)
    results_all["baseline_lof"] = binary_anomaly_metrics(y_binary_true, lof_scores, y_pred_lof)
    print(f"  Baseline LOF F1: {results_all['baseline_lof']['f1']:.4f}")

    # Baseline 3: IF on sparse checkpoint features only (no blockchain, no sketch)
    sparse_features = X_test[:, :9]  # checkpoint features only
    sparse_train = X_train[:, :9]
    baseline_sparse = BaselineDetector(
        contamination=config["detection"]["isolation_forest"]["contamination"],
        random_state=seed,
    )
    baseline_sparse.fit(sparse_train)
    y_pred_sparse = baseline_sparse.predict(sparse_features)
    sparse_scores = -baseline_sparse.decision_function(sparse_features)
    results_all["baseline_sparse_if"] = binary_anomaly_metrics(
        y_binary_true, sparse_scores, y_pred_sparse
    )
    print(f"  Baseline Sparse IF F1: {results_all['baseline_sparse_if']['f1']:.4f}")

    # Baseline 4: IF on sketch features only (no checkpoints, simulating blockchain-only)
    sketch_features = X_test[:, 9:18]
    sketch_train = X_train[:, 9:18]
    baseline_sketch = BaselineDetector(
        contamination=config["detection"]["isolation_forest"]["contamination"],
        random_state=seed,
    )
    baseline_sketch.fit(sketch_train)
    y_pred_sketch = baseline_sketch.predict(sketch_features)
    sketch_scores = -baseline_sketch.decision_function(sketch_features)
    results_all["baseline_sketch_if"] = binary_anomaly_metrics(
        y_binary_true, sketch_scores, y_pred_sketch
    )
    print(f"  Baseline Sketch IF F1: {results_all['baseline_sketch_if']['f1']:.4f}")

    # New baselines: OC-SVM, TRAOD, iBAT
    print("\n  Running OC-SVM baseline...")
    ocsvm = BaselineOCSVM(nu=0.05)
    ocsvm.fit(X_train)
    y_pred_ocsvm = ocsvm.predict(X_test)
    ocsvm_scores = -ocsvm.decision_function(X_test)
    results_all["baseline_ocsvm"] = binary_anomaly_metrics(
        y_binary_true, ocsvm_scores, y_pred_ocsvm
    )
    print(f"  Baseline OC-SVM F1: {results_all['baseline_ocsvm']['f1']:.4f}")

    print("  Running TRAOD baseline...")
    traod = BaselineTRAOD(min_neighbors=5)
    traod.fit(X_train)
    y_pred_traod = traod.predict(X_test)
    traod_scores = -traod.decision_function(X_test)
    results_all["baseline_traod"] = binary_anomaly_metrics(
        y_binary_true, traod_scores, y_pred_traod
    )
    print(f"  Baseline TRAOD F1: {results_all['baseline_traod']['f1']:.4f}")

    print("  Running iBAT baseline...")
    ibat = BaselineIBAT(n_bins=10, min_support=3)
    ibat.fit(X_train)
    y_pred_ibat = ibat.predict(X_test)
    ibat_scores = -ibat.decision_function(X_test)
    results_all["baseline_ibat"] = binary_anomaly_metrics(
        y_binary_true, ibat_scores, y_pred_ibat
    )
    print(f"  Baseline iBAT F1: {results_all['baseline_ibat']['f1']:.4f}")

    # Attack coverage analysis
    print("\n[6/6] Attack coverage analysis...")
    attack_coverage = attack_coverage_report(test_attack_types, y_test, y_pred_wp)
    results_all["attack_coverage"] = attack_coverage
    for atype, stats in attack_coverage.items():
        if atype != "none":
            print(f"  {atype}: {stats['detection_rate']:.1%} detected "
                  f"({stats['detected']}/{stats['total']})")

    # Blockchain metrics
    bc_metrics = blockchain_metrics_summary(contract.get_metrics())
    results_all["blockchain_metrics"] = bc_metrics
    print(f"\n  Blockchain storage/commit: {bc_metrics['storage_per_commitment_bytes']:.0f} bytes")
    print(f"  Total gas cost: {bc_metrics['total_gas_cost']:.0f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")

    serializable = {}
    for k, v in results_all.items():
        if isinstance(v, dict):
            serializable[k] = {
                sk: sv if not isinstance(sv, np.ndarray) else sv.tolist()
                for sk, sv in v.items()
            }
        else:
            serializable[k] = v

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results_all


def run_ablation_study(config: dict, output_dir: str):
    """Run ablation study: remove components one at a time."""
    print("\n" + "=" * 60)
    print("Ablation Study")
    print("=" * 60)

    seed = config["experiment"]["random_seed"]
    np.random.seed(seed)

    trajectories = generate_synthetic_trajectories(
        n_trajectories=500, min_length=20, max_length=80, seed=seed
    )
    n_test = int(len(trajectories) * config["experiment"]["test_ratio"])
    train_trajs = trajectories[:-n_test]
    test_trajs = trajectories[-n_test:]

    capture = SecureCaptureModule(
        device_id="ablation", window_size_sec=config["data"]["window_size_sec"]
    )
    contract = WindowChainContract(
        min_checkpoint_density=config["checkpoint"]["min_checkpoint_density"]
    )
    chain = WindowChain(contract)
    attacker = AttackSimulator(seed=seed)

    # Generate training features
    train_features = []
    for traj in train_trajs:
        records = process_trajectory_through_pipeline(traj, capture, chain, "ablation")
        for r in records:
            train_features.append(r["features"])
    X_train = np.array(train_features)

    # Generate test features with attacks
    test_features, test_labels = [], []
    for traj in test_trajs:
        if np.random.random() < config["experiment"]["anomaly_ratio"]:
            attacked, info = attacker.generate_random_attack(traj)
            cap2 = SecureCaptureModule(device_id="abl_test", window_size_sec=config["data"]["window_size_sec"])
            orig_wins = capture.process_trajectory(traj)
            atk_wins = cap2.process_trajectory(attacked)
            label = 1 if info["type"] in ["point_deletion", "point_injection", "timestamp_shift", "replay"] else 2
            for j in range(min(len(orig_wins), len(atk_wins))):
                committed_sketch = orig_wins[j]["sketch"] if label == 1 else atk_wins[j]["sketch"]
                cp_pts = [atk_wins[j]["raw_points"][k] for k in atk_wins[j]["checkpoint_indices"]
                          if k < len(atk_wins[j]["raw_points"])]
                cp_feats = compute_checkpoint_features(atk_wins[j]["raw_points"], atk_wins[j]["checkpoint_indices"])
                recon = reconstruct_sketch_from_checkpoints(cp_pts)
                resid = compute_sketch_residual(committed_sketch, recon)
                flags = {"window_missing": False, "late_commitment": False,
                         "proof_failed": label == 1, "density_violation": False}
                feat = build_detection_features(cp_feats, committed_sketch, resid, flags)
                test_features.append(feat)
                test_labels.append(label)
        else:
            records = process_trajectory_through_pipeline(traj, capture, chain, "abl_test")
            for r in records:
                test_features.append(r["features"])
                test_labels.append(0)

    X_test = np.array(test_features)
    y_test = np.array(test_labels)

    ablation_configs = {
        "full_model": list(range(24)),
        "no_sketch_residual": list(range(18)) + [20, 21, 22, 23],  # remove indices 18,19
        "no_integrity_flags": list(range(20)),                       # remove indices 20-23
        "no_committed_sketch": list(range(9)) + list(range(18, 24)), # remove indices 9-17
        "checkpoint_only": list(range(9)),                           # only checkpoint features
        "sketch_only": list(range(9, 18)),                           # only sketch features
    }

    ablation_results = {}
    for name, feature_mask in ablation_configs.items():
        X_tr_masked = X_train[:, feature_mask]
        X_te_masked = X_test[:, feature_mask]

        det = BaselineDetector(
            contamination=config["detection"]["isolation_forest"]["contamination"],
            random_state=seed,
        )
        det.fit(X_tr_masked)
        y_pred = det.predict(X_te_masked)
        y_binary = (y_test > 0).astype(int)
        scores = -det.decision_function(X_te_masked)
        metrics = binary_anomaly_metrics(y_binary, scores, y_pred)
        ablation_results[name] = metrics
        print(f"  {name}: F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}")

    abl_path = os.path.join(output_dir, "ablation_results.json")
    with open(abl_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAblation results saved to {abl_path}")

    return ablation_results


def run_checkpoint_density_sweep(config: dict, output_dir: str):
    """Sweep checkpoint density to study detection vs. cost tradeoff."""
    print("\n" + "=" * 60)
    print("Checkpoint Density Sweep")
    print("=" * 60)

    seed = config["experiment"]["random_seed"]
    np.random.seed(seed)

    trajectories = generate_synthetic_trajectories(
        n_trajectories=300, min_length=30, max_length=80, seed=seed
    )

    densities = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    sweep_results = {}

    for density in densities:
        cfg_copy = dict(config)
        cfg_copy["checkpoint"] = dict(config["checkpoint"])
        cfg_copy["checkpoint"]["min_checkpoint_density"] = density

        capture = SecureCaptureModule(
            device_id="sweep", window_size_sec=config["data"]["window_size_sec"]
        )
        contract = WindowChainContract(min_checkpoint_density=density)
        chain = WindowChain(contract)

        features, labels = [], []
        attacker = AttackSimulator(seed=seed)

        for traj in trajectories:
            if np.random.random() < 0.1:
                attacked, info = attacker.generate_random_attack(traj, category="integrity")
                label = 1
            else:
                attacked = traj
                label = 0

            records = process_trajectory_through_pipeline(attacked, capture, chain, "sweep")
            for r in records:
                features.append(r["features"])
                labels.append(label)

        X = np.array(features)
        y = np.array(labels)

        n_tr = int(len(X) * 0.7)
        det = BaselineDetector(contamination=0.05, random_state=seed)
        det.fit(X[:n_tr])
        y_pred = det.predict(X[n_tr:])
        scores = -det.decision_function(X[n_tr:])
        metrics = binary_anomaly_metrics(y[n_tr:], scores, y_pred)

        bc_metrics = contract.get_metrics()
        sweep_results[str(density)] = {
            **metrics,
            "storage_bytes": bc_metrics["total_storage_bytes"],
            "gas_cost": bc_metrics["estimated_gas_cost"],
        }
        print(f"  Density {density:.2f}: F1={metrics['f1']:.4f}, "
              f"Storage={bc_metrics['total_storage_bytes']} bytes")

    sweep_path = os.path.join(output_dir, "density_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to {sweep_path}")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WindowProof Experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--data", default="synthetic",
                        choices=["synthetic", "porto", "tdrive"],
                        help="Data source: synthetic, porto (real), or tdrive (real)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--sweep", action="store_true", help="Run density sweep")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    run_main = args.all or (not args.ablation and not args.sweep)
    if run_main or args.all:
        run_experiment(config, args.output, args.data)

    if args.ablation or args.all:
        run_ablation_study(config, args.output)

    if args.sweep or args.all:
        run_checkpoint_density_sweep(config, args.output)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
