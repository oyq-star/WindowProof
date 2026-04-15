"""Download and prepare real trajectory datasets."""

import os
import sys
import ssl
import zipfile
import urllib.request
import shutil
from pathlib import Path

# Workaround for SSL certificate issues on some systems
ssl._create_default_https_context = ssl._create_unverified_context


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress display."""
    print(f"Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest, _progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\nFailed to download: {e}")
        return False


def _progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
    else:
        mb = count * block_size / (1024 * 1024)
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


def download_tdrive():
    """Download T-Drive dataset from Microsoft Research.

    Contains ~10,357 taxis in Beijing, 1 week of data.
    Format per file: taxi_id, datetime, longitude, latitude
    """
    tdrive_dir = DATA_DIR / "tdrive"
    tdrive_dir.mkdir(parents=True, exist_ok=True)

    # T-Drive dataset zip files from Microsoft Research
    base_url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02"
    zip_files = [
        "06.zip", "07.zip", "08.zip", "09.zip", "010.zip",
        "011.zip", "012.zip", "013.zip", "014.zip",
    ]

    print("=" * 50)
    print("Downloading T-Drive Dataset (Beijing Taxi)")
    print("=" * 50)

    for zf in zip_files:
        url = f"{base_url}/{zf}"
        dest = tdrive_dir / zf
        if dest.exists():
            print(f"  {zf} already exists, skipping.")
            continue

        success = download_file(url, str(dest), zf)
        if success:
            print(f"  Extracting {zf}...")
            try:
                with zipfile.ZipFile(str(dest), "r") as z:
                    z.extractall(str(tdrive_dir))
                print(f"  Done: {zf}")
            except zipfile.BadZipFile:
                print(f"  Bad zip file: {zf}, removing.")
                dest.unlink()

    # Count extracted files
    txt_count = len(list(tdrive_dir.rglob("*.txt")))
    print(f"\nT-Drive: {txt_count} trajectory files ready in {tdrive_dir}")
    return str(tdrive_dir)


def download_porto():
    """Download Porto Taxi dataset.

    Contains 1.7M trajectories over 1 year.
    We try Figshare first, then provide manual instructions.
    """
    porto_dir = DATA_DIR / "porto"
    porto_dir.mkdir(parents=True, exist_ok=True)
    porto_csv = porto_dir / "train.csv"

    if porto_csv.exists():
        print(f"Porto dataset already exists at {porto_csv}")
        return str(porto_csv)

    print("=" * 50)
    print("Porto Taxi Dataset")
    print("=" * 50)

    # Try Figshare download
    figshare_url = "https://figshare.com/ndownloader/files/22677902"
    dest = porto_dir / "porto_taxi.csv"
    success = download_file(figshare_url, str(dest), "Porto taxi from Figshare")

    if success and dest.exists() and dest.stat().st_size > 1000:
        dest.rename(porto_csv)
        print(f"Porto dataset ready at {porto_csv}")
        return str(porto_csv)

    # If download fails, provide manual instructions
    print("\n" + "=" * 50)
    print("MANUAL DOWNLOAD REQUIRED for Porto dataset:")
    print("=" * 50)
    print("Option 1: UCI ML Repository")
    print("  https://archive.ics.uci.edu/dataset/339/")
    print("  Download and place train.csv in:", porto_dir)
    print()
    print("Option 2: Kaggle (requires account)")
    print("  https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data")
    print("  Download train.csv.zip, extract, place in:", porto_dir)
    print()
    print("Option 3: Figshare")
    print("  https://figshare.com/articles/dataset/Porto_taxi_trajectories/12302165")
    print("  Download and rename to train.csv in:", porto_dir)
    print("=" * 50)

    return None


def prepare_sample_from_tdrive(tdrive_dir: str, output_path: str,
                                max_files: int = 200, min_points: int = 20):
    """Prepare a preprocessed sample from T-Drive for quick experiments."""
    import csv
    from datetime import datetime

    print(f"\nPreparing T-Drive sample (max {max_files} taxis)...")
    tdrive_path = Path(tdrive_dir)

    txt_files = sorted(tdrive_path.rglob("*.txt"))[:max_files]
    trajectories = []

    for fpath in txt_files:
        points = []
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    taxi_id = parts[0].strip()
                    dt = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
                    lon = float(parts[2].strip())
                    lat = float(parts[3].strip())
                    # Basic filter: Beijing area
                    if 39.4 < lat < 41.0 and 115.5 < lon < 117.5:
                        points.append((lat, lon, dt.timestamp(), taxi_id))
                except (ValueError, IndexError):
                    continue

        if len(points) >= min_points:
            points.sort(key=lambda x: x[2])
            # Split into trips by time gap > 30 min
            trips = []
            current_trip = [points[0]]
            for i in range(1, len(points)):
                if points[i][2] - points[i - 1][2] > 1800:
                    if len(current_trip) >= min_points:
                        trips.append(current_trip)
                    current_trip = [points[i]]
                else:
                    current_trip.append(points[i])
            if len(current_trip) >= min_points:
                trips.append(current_trip)

            trajectories.extend(trips)

    # Save as CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["traj_id", "point_idx", "lat", "lon", "timestamp"])
        for tid, traj in enumerate(trajectories):
            for pid, (lat, lon, ts, _) in enumerate(traj):
                writer.writerow([tid, pid, f"{lat:.8f}", f"{lon:.8f}", f"{ts:.3f}"])

    print(f"  Saved {len(trajectories)} trajectories to {output}")
    return len(trajectories)


def download_geolife():
    """Download GeoLife GPS trajectory dataset from Microsoft Research.

    Contains 17,621 trajectories from 182 users in Beijing, 2007-2012.
    Multi-modal: walking, driving, bus, bike, etc.
    Format per .plt file: lat, lon, 0, altitude, date_days, date, time
    """
    geolife_dir = DATA_DIR / "geolife"
    geolife_dir.mkdir(parents=True, exist_ok=True)

    zip_path = geolife_dir / "geolife.zip"
    extracted_marker = geolife_dir / "Data"

    if extracted_marker.exists():
        print(f"GeoLife dataset already extracted in {geolife_dir}")
        return str(geolife_dir)

    print("=" * 50)
    print("Downloading GeoLife Dataset (Multi-modal GPS)")
    print("=" * 50)

    url = ("https://download.microsoft.com/download/F/4/8/"
           "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/"
           "Geolife%20Trajectories%201.3.zip")
    success = download_file(url, str(zip_path), "GeoLife Trajectories 1.3")

    if success and zip_path.exists():
        print("  Extracting (this may take a minute)...")
        try:
            with zipfile.ZipFile(str(zip_path), "r") as z:
                z.extractall(str(geolife_dir))
            print("  Done.")
        except zipfile.BadZipFile:
            print("  Bad zip file, removing.")
            zip_path.unlink()
            return None

    plt_count = len(list(geolife_dir.rglob("*.plt")))
    print(f"\nGeoLife: {plt_count} trajectory files ready in {geolife_dir}")
    return str(geolife_dir)


def prepare_sample_from_geolife(geolife_dir: str, output_path: str,
                                 max_users: int = 50, min_points: int = 20):
    """Prepare a preprocessed sample from GeoLife for quick experiments."""
    import csv
    from datetime import datetime

    print(f"\nPreparing GeoLife sample (max {max_users} users)...")
    geolife_path = Path(geolife_dir)

    # Find the Data directory (may be nested)
    data_root = geolife_path / "Geolife Trajectories 1.3" / "Data"
    if not data_root.exists():
        data_root = geolife_path / "Data"
    if not data_root.exists():
        # Try to find any directory with numbered user folders
        for d in geolife_path.rglob("000"):
            data_root = d.parent
            break

    if not data_root.exists():
        print(f"  Could not find GeoLife Data directory in {geolife_dir}")
        return 0

    user_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])[:max_users]
    trajectories = []

    for user_dir in user_dirs:
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue

        plt_files = sorted(traj_dir.glob("*.plt"))
        for fpath in plt_files:
            points = []
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                # Skip 6 header lines
                for _ in range(6):
                    next(f, None)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 7:
                        continue
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        # parts[5] = date, parts[6] = time
                        dt_str = f"{parts[5]} {parts[6]}"
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        # Basic filter: Beijing area
                        if 39.4 < lat < 41.0 and 115.5 < lon < 117.5:
                            points.append((lat, lon, dt.timestamp()))
                    except (ValueError, IndexError, StopIteration):
                        continue

            if len(points) >= min_points:
                points.sort(key=lambda x: x[2])
                # Split by time gap > 20 min (multi-modal trips are shorter)
                trips = []
                current_trip = [points[0]]
                for i in range(1, len(points)):
                    if points[i][2] - points[i - 1][2] > 1200:
                        if len(current_trip) >= min_points:
                            trips.append(current_trip)
                        current_trip = [points[i]]
                    else:
                        current_trip.append(points[i])
                if len(current_trip) >= min_points:
                    trips.append(current_trip)

                trajectories.extend(trips)

    # Save as CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["traj_id", "point_idx", "lat", "lon", "timestamp"])
        for tid, traj in enumerate(trajectories):
            for pid, (lat, lon, ts) in enumerate(traj):
                writer.writerow([tid, pid, f"{lat:.8f}", f"{lon:.8f}", f"{ts:.3f}"])

    print(f"  Saved {len(trajectories)} trajectories to {output}")
    return len(trajectories)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download trajectory datasets")
    parser.add_argument("--dataset", choices=["tdrive", "porto", "geolife", "all"],
                        default="all")
    args = parser.parse_args()

    if args.dataset in ("tdrive", "all"):
        tdrive_dir = download_tdrive()
        if tdrive_dir:
            sample_path = str(DATA_DIR / "tdrive_sample.csv")
            prepare_sample_from_tdrive(tdrive_dir, sample_path)

    if args.dataset in ("geolife", "all"):
        geolife_dir = download_geolife()
        if geolife_dir:
            sample_path = str(DATA_DIR / "geolife_sample.csv")
            prepare_sample_from_geolife(geolife_dir, sample_path)

    if args.dataset in ("porto", "all"):
        download_porto()
