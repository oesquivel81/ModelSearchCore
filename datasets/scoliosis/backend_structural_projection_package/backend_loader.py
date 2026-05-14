
"""
Backend loader for structural projection shards.

Expected package structure:

BACKEND_STRUCTURAL_PROJECTION_PACKAGE/
  data/
    shards_npz/
    patient_images/
    tables/
  config/
    backend_structural_projection_bundle.joblib
  models/
    best_model_checkpoint.pt

NPZ fields:
  X: [N, C, H, W] float32
  patient_key: [N]
  channel_names: [C]
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd


def load_backend_bundle(package_root):
    bundle_path = os.path.join(
        package_root,
        "config",
        "backend_structural_projection_bundle.joblib"
    )
    return joblib.load(bundle_path)


def list_shards(package_root):
    shard_dir = os.path.join(package_root, "data", "shards_npz")
    return sorted(glob.glob(os.path.join(shard_dir, "*.npz")))


def load_shard(shard_path):
    data = np.load(shard_path, allow_pickle=True)

    return {
        "X": data["X"].astype(np.float32),
        "patient_key": data["patient_key"].astype(str),
        "channel_names": data["channel_names"].astype(str),
        "status": data["status"].astype(str) if "status" in data.files else None,
        "pred_regions": data["pred_regions"].astype(str) if "pred_regions" in data.files else None,
        "n_pred_regions": data["n_pred_regions"].astype(str) if "n_pred_regions" in data.files else None,
    }


def load_all_shards(package_root):
    shards = list_shards(package_root)

    X_parts = []
    patient_parts = []
    status_parts = []

    channel_names = None

    for shard_path in shards:
        shard = load_shard(shard_path)

        X_parts.append(shard["X"])
        patient_parts.append(shard["patient_key"])

        if shard["status"] is not None:
            status_parts.append(shard["status"])

        if channel_names is None:
            channel_names = shard["channel_names"]

    X = np.concatenate(X_parts, axis=0)
    patient_key = np.concatenate(patient_parts, axis=0)

    status = None
    if len(status_parts) > 0:
        status = np.concatenate(status_parts, axis=0)

    return {
        "X": X,
        "patient_key": patient_key,
        "status": status,
        "channel_names": channel_names,
    }


def load_manifest(package_root):
    manifest_path = os.path.join(
        package_root,
        "data",
        "tables",
        "structural_shards_tables",
        "structural_projection_manifest.csv"
    )

    if not os.path.exists(manifest_path):
        return None

    return pd.read_csv(manifest_path)


def get_patient_image_paths(package_root, patient_key):
    pdir = os.path.join(
        package_root,
        "data",
        "patient_images",
        str(patient_key)
    )

    return {
        "patient_dir": pdir,
        "baseline": os.path.join(pdir, "normalized_full_image_used.png"),
        "overlay": os.path.join(pdir, "overlay_pred_regions_curve.png"),
        "binary": os.path.join(pdir, "pred_binary_confidence.png"),
        "boundary": os.path.join(pdir, "pred_boundary.png"),
        "intervertebral": os.path.join(pdir, "pred_intervertebral.png"),
        "ordinal": os.path.join(pdir, "pred_ordinal.png"),
        "summary": os.path.join(pdir, "reconstruction_summary.json"),
    }


if __name__ == "__main__":
    package_root = os.path.dirname(os.path.abspath(__file__))

    bundle = load_backend_bundle(package_root)
    print("Loaded bundle:", bundle["package_name"])
    print("Channels:", bundle["channels"])

    shards = list_shards(package_root)
    print("N shards:", len(shards))

    if shards:
        s0 = load_shard(shards[0])
        print("First shard X:", s0["X"].shape)
        print("First shard patients:", s0["patient_key"][:5])
