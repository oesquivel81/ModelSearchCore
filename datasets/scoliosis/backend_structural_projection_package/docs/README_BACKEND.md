# BACKEND_STRUCTURAL_PROJECTION_PACKAGE

Backend-ready package for scoliosis structural projection inference.

## Contents

Directory structure:

    BACKEND_STRUCTURAL_PROJECTION_PACKAGE/
      backend_loader.py
      data/
        shards_npz/
          structural_projection_shard_00000.npz
          ...
        patient_images/
          <patient_key>/
            normalized_full_image_used.png
            overlay_pred_regions_curve.png
            pred_binary_confidence.png
            pred_boundary.png
            pred_intervertebral.png
            pred_ordinal.png
            reconstruction_summary.json
            pasted_patches.csv
        tables/
          structural_shards_tables/
          patient_reconstruction_tables/
        validation_grids/
      models/
        best_model_checkpoint.pt
      config/
        backend_structural_projection_bundle.joblib
      docs/
        README_BACKEND.md

## Structural channels

The NPZ shards contain:

    X.shape == [N, C, H, W]

Channels:

    CH0 baseline
    CH1 binary
    CH2 boundary
    CH3 intervertebral
    CH4 ordinal

These are intended as structural inputs for a second-stage anatomical CNN
or backend inference pipeline.

## Quick load

    from backend_loader import load_all_shards, load_backend_bundle

    package_root = "/path/to/BACKEND_STRUCTURAL_PROJECTION_PACKAGE"

    bundle = load_backend_bundle(package_root)
    data = load_all_shards(package_root)

    X = data["X"]
    patient_key = data["patient_key"]
    channel_names = data["channel_names"]

    print(X.shape)
    print(channel_names)

## Checkpoint

Model checkpoint path:

    models/best_model_checkpoint.pt

## Created

2026-05-14T19:26:45.162418Z
