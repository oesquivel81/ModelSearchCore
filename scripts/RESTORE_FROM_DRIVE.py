
# ============================================================
# RESTORE — RESTAURAR BACKUP DESDE DRIVE
# ============================================================

import os, tarfile, shutil, json
from pathlib import Path
import pandas as pd

DRIVE_ARCHIVE_PATH = Path(r"/content/drive/MyDrive/TDA_PIPELINE/BACKUPS_VERTEBRA_PEAK_ANALYSIS/SCOLIOSIS_VERTEBRA_PEAK_ANALYSIS_20260515_200103.tar.gz")
RESTORE_ROOT = Path("/content")

assert DRIVE_ARCHIVE_PATH.exists(), f"No existe el backup: {DRIVE_ARCHIVE_PATH}"

print("Restaurando:", DRIVE_ARCHIVE_PATH)

with tarfile.open(DRIVE_ARCHIVE_PATH, "r:gz") as tar:
    tar.extractall(RESTORE_ROOT)

RESTORED_DIR = RESTORE_ROOT / "SCOLIOSIS_VERTEBRA_PEAK_ANALYSIS_20260515_200103"

print("Restaurado en:", RESTORED_DIR)

BASE_ROOT = Path("/content/PATIENT_RECONSTRUCTED_PREDICTIONS_FROM_PATCHES")
PATIENTS_ROOT = BASE_ROOT / "patients"

BASE_ROOT.mkdir(parents=True, exist_ok=True)
PATIENTS_ROOT.mkdir(parents=True, exist_ok=True)

global_outputs = RESTORED_DIR / "global_outputs"

if global_outputs.exists():
    for child in global_outputs.iterdir():
        if not child.is_dir():
            continue

        dst = BASE_ROOT / child.name

        if child.name == "RADIOGRAPH_GEOMETRY_METRICS":
            dst = Path("/content/RADIOGRAPH_GEOMETRY_METRICS")
        elif child.name == "patient_spatial_csvs":
            dst = Path("/content/patient_spatial_csvs")

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(child, dst)
        print("Copiado:", child, "->", dst)

patients_minimal = RESTORED_DIR / "patients_minimal"

if patients_minimal.exists():
    for pdir in patients_minimal.iterdir():
        if pdir.is_dir():
            dst = PATIENTS_ROOT / pdir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(pdir, dst)

    print("Pacientes restaurados:", len(list(PATIENTS_ROOT.glob("*_curve_masked"))))

visible_csv = BASE_ROOT / "BODY_VERTEBRA_VISIBLE_ORDER_CLEAN" / "BODY_VERTEBRA_VISIBLE_ORDER_DEDUPED.csv"
if visible_csv.exists():
    df_visible_clean = pd.read_csv(visible_csv)
    print("df_visible_clean:", df_visible_clean.shape)
    display(df_visible_clean.head())

print("RESTORE terminado.")
