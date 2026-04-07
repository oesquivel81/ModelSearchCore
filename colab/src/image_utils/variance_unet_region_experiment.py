"""
Experimento de segmentación UNet con varianza sobre regiones vertebrales
extraídas por VertebraComponentExtractor.

Flujo:
  1. Lee dataset_index.csv y separa train / val / test a nivel de imagen.
  2. Extrae regiones vertebrales con VertebraRegionBatch para cada split.
  3. Agrupa imágenes por número de vértebras detectadas.
  4. Entrena la UNet sobre los parches (no sobre imágenes completas).
  5. Guarda métricas, plots y checkpoints en Drive.

Config de ejemplo:
    config_dict = {
        "experiment_name": "unet_variance_regions_v1",
        "drive_root": "/content/drive/MyDrive/.../cnn_varianza_a",
        "seed": 42,
        "execution_mode": "variance_input",

        "extractor": {
            "base_dir": "/content/drive/MyDrive/.../MaIA_Scoliosis_Dataset",
            "index_csv": "/content/drive/MyDrive/.../dataset_index.csv",
            "image_col": "radiograph_path",
            "mask_col": "label_binary_path",
            "min_area": 150,
            "pad_x": 30,
            "pad_y": 15,
            "top_region_ratio": 0.35,
            "top_pad_x_scale": 2,
            "top_pad_y_top_scale": 3,
            "top_pad_y_bottom_scale": 0.8,
            "pad_x_tight": 8,
            "pad_y_tight": 6,
            "top_extra_tight": 10,
            "mode": "tight",
            "include_labels": ["good"]
        },

        "data": {
            "patch_size": [128, 128],
            "binarize_mask": True
        },

        "model": {
            "type": "variance_input",
            "base_channels": 32,
            "model_name": "unet"
        },

        "training": {
            "batch_size": 8,
            "epochs": 20,
            "lr": 1e-3,
            "num_workers": 2,
            "save_visuals_each_epoch": True,
            "num_visual_samples": 3,
            "best_metric_name": "dice",
            "resume_checkpoint_path": None
        }
    }
"""

import os
import json
import time
import shutil
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import directed_hausdorff
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .vertebra_region_batch import VertebraRegionBatch
from .vertebra_region_dataset import VertebraRegionDataset

try:
    from varianza_patches_cnn.discord_webhook_notifier import DiscordWebhookNotifier
except ImportError:
    DiscordWebhookNotifier = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import requests as _requests
except ImportError:
    _requests = None


# =========================================================
# UTILIDADES
# =========================================================
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, data: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def normalize_split_value(x: str) -> str:
    x = str(x).strip().lower()
    split_map = {
        "training": "train", "train": "train", "tr": "train",
        "validation": "val", "valid": "val", "val": "val", "dev": "val",
        "testing": "test", "test": "test", "ts": "test",
    }
    return split_map.get(x, x)


# =========================================================
# MÉTRICAS
# =========================================================
def dice_from_probs(preds, targets, eps=1e-8):
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_from_probs(preds, targets, eps=1e-8):
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def precision_from_probs(preds, targets, eps=1e-8):
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fp + eps)).mean().item()


def recall_from_probs(preds, targets, eps=1e-8):
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fn + eps)).mean().item()


def f1_from_precision_recall(precision, recall, eps=1e-8):
    return (2 * precision * recall) / (precision + recall + eps)


def hausdorff_distance_binary(pred, target):
    pred_pts = np.argwhere(pred > 0)
    tgt_pts = np.argwhere(target > 0)
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return np.nan
    d1 = directed_hausdorff(pred_pts, tgt_pts)[0]
    d2 = directed_hausdorff(tgt_pts, pred_pts)[0]
    return max(d1, d2)


def get_disk_free_gb(path="."):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def get_system_metrics():
    metrics = {
        "cpu_percent": None, "ram_used_mb": None, "ram_total_mb": None,
        "ram_percent": None, "gpu_name": None, "gpu_memory_used_mb": None,
        "gpu_memory_reserved_mb": None, "gpu_memory_total_mb": None,
        "gpu_utilization_percent": None, "disk_free_gb": None,
    }

    if psutil is not None:
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.2)
            vm = psutil.virtual_memory()
            metrics["ram_used_mb"] = vm.used / (1024 ** 2)
            metrics["ram_total_mb"] = vm.total / (1024 ** 2)
            metrics["ram_percent"] = vm.percent
        except Exception:
            pass

    try:
        metrics["disk_free_gb"] = get_disk_free_gb(".")
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["gpu_memory_used_mb"] = torch.cuda.memory_allocated(0) / (1024 ** 2)
            metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / (1024 ** 2)
            props = torch.cuda.get_device_properties(0)
            metrics["gpu_memory_total_mb"] = props.total_memory / (1024 ** 2)
        except Exception:
            pass
        metrics["gpu_utilization_percent"] = None

    return metrics


# =========================================================
# CAPAS Y MODELOS
# =========================================================
class LocalVarianceLayer(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        mean_sq = F.avg_pool2d(x * x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return torch.clamp(mean_sq - mean * mean, min=0.0)


class VarianceInputLayer(nn.Module):
    def __init__(self, kernel_sizes=(3, 5, 9)):
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        outs = [x]
        for k in self.kernel_sizes:
            mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
            mean_sq = F.avg_pool2d(x * x, kernel_size=k, stride=1, padding=k // 2)
            outs.append(torch.clamp(mean_sq - mean * mean, min=0.0))
        return torch.cat(outs, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetBaseline(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.dec1 = DecoderBlock(base * 2, base, base)
        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        s1, x1 = self.enc1(x)
        s2, x2 = self.enc2(x1)
        s3, x3 = self.enc3(x2)
        b = self.bottleneck(x3)
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1)


class UNetVarianceInput(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.var_input = VarianceInputLayer(kernel_sizes=(3, 5, 9))
        total_in = in_channels * 4
        self.enc1 = EncoderBlock(total_in, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.dec1 = DecoderBlock(base * 2, base, base)
        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.var_input(x)
        s1, x1 = self.enc1(x)
        s2, x2 = self.enc2(x1)
        s3, x3 = self.enc3(x2)
        b = self.bottleneck(x3)
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1)


class UNetVarianceBranch(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.var_layer = LocalVarianceLayer(kernel_size=5)
        self.img_enc1 = EncoderBlock(in_channels, base)
        self.img_enc2 = EncoderBlock(base, base * 2)
        self.var_enc1 = EncoderBlock(in_channels, base)
        self.var_enc2 = EncoderBlock(base, base * 2)
        self.fuse_enc3 = EncoderBlock(base * 4, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 4, base * 2)
        self.dec1 = DecoderBlock(base * 2, base * 2, base)
        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        v = self.var_layer(x)
        s1_img, x1_img = self.img_enc1(x)
        s2_img, x2_img = self.img_enc2(x1_img)
        s1_var, x1_var = self.var_enc1(v)
        s2_var, x2_var = self.var_enc2(x1_var)
        x_fused = torch.cat([x2_img, x2_var], dim=1)
        s_fused = torch.cat([s2_img, s2_var], dim=1)
        s3, x3 = self.fuse_enc3(x_fused)
        b = self.bottleneck(x3)
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s_fused)
        d1 = self.dec1(d2, torch.cat([s1_img, s1_var], dim=1))
        return self.out_conv(d1)


# =========================================================
# EXPERIMENTO SOBRE REGIONES VERTEBRALES
# =========================================================
class VarianceUNetRegionExperiment:
    """
    Entrena una UNet (baseline / variance_input / variance_branch)
    sobre parches de regiones vertebrales extraídas con
    VertebraComponentExtractor, agrupados por número de vértebras.
    """

    def __init__(self, config: dict):
        self.config = config
        self.experiment_name = config.get("experiment_name", "unet_region_experiment")
        self.drive_root = config.get("drive_root", "./results")
        self.seed = int(config.get("seed", 42))
        self.execution_mode = config.get("execution_mode", "variance_input")

        # --- Extractor (usa base_dir e index_csv) ---
        ext = config.get("extractor", {})
        self.base_dir = ext["base_dir"]
        self.index_csv = ext["index_csv"]
        self.image_col = ext.get("image_col", "radiograph_path")
        self.mask_col = ext.get("mask_col", "label_binary_path")
        self.extractor_params = {
            "min_area": int(ext.get("min_area", 150)),
            "pad_x": int(ext.get("pad_x", 30)),
            "pad_y": int(ext.get("pad_y", 15)),
            "top_region_ratio": float(ext.get("top_region_ratio", 0.35)),
            "top_pad_x_scale": float(ext.get("top_pad_x_scale", 2)),
            "top_pad_y_top_scale": float(ext.get("top_pad_y_top_scale", 3)),
            "top_pad_y_bottom_scale": float(ext.get("top_pad_y_bottom_scale", 0.8)),
            "pad_x_tight": int(ext.get("pad_x_tight", 8)),
            "pad_y_tight": int(ext.get("pad_y_tight", 6)),
            "top_extra_tight": int(ext.get("top_extra_tight", 10)),
            "mode": ext.get("mode", "tight"),
            "include_labels": ext.get("include_labels", ["good"]),
        }

        # --- Data (CNN) ---
        data_cfg = config.get("data", {})
        self.patch_size = tuple(data_cfg.get("patch_size", [128, 128]))
        self.binarize_mask = bool(data_cfg.get("binarize_mask", True))

        # --- Training ---
        tr = config.get("training", {})
        self.batch_size = int(tr.get("batch_size", 8))
        self.epochs = int(tr.get("epochs", 20))
        self.lr = float(tr.get("lr", 1e-3))
        self.num_workers = int(tr.get("num_workers", 2))
        self.save_visuals_each_epoch = bool(tr.get("save_visuals_each_epoch", True))
        self.num_visual_samples = int(tr.get("num_visual_samples", 3))
        self.best_metric_name = tr.get("best_metric_name", "dice")
        self.resume_checkpoint_path = tr.get("resume_checkpoint_path", None)

        # --- Model ---
        mc = config.get("model", {})
        self.model_type = mc.get("type", self.execution_mode)
        self.base_channels = int(mc.get("base_channels", 32))
        self.model_name = mc.get("model_name", "unet")

        # --- Discord ---
        dc = config.get("discord", {})
        self.discord_webhook_url = dc.get("webhook_url", None)
        self.discord_notify_every = int(dc.get("notify_every_n_epochs", 1))

        if self.discord_webhook_url and DiscordWebhookNotifier is not None:
            self.notifier = DiscordWebhookNotifier(
                webhook_url=self.discord_webhook_url,
                experiment_name=self.experiment_name,
            )
        else:
            self.notifier = None

        # --- Device ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Directorios de salida ---
        self.output_dir = os.path.join(self.drive_root, self.experiment_name)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.epoch_samples_dir = os.path.join(self.output_dir, "epoch_samples")
        self.filters_dir = os.path.join(self.output_dir, "filters")
        self.best_predictions_dir = os.path.join(self.output_dir, "best_predictions")
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        self.groups_dir = os.path.join(self.output_dir, "groups")

        for d in [self.output_dir, self.plots_dir, self.epoch_samples_dir,
                  self.filters_dir, self.best_predictions_dir,
                  self.checkpoints_dir, self.groups_dir]:
            ensure_dir(d)

        # --- Estado ---
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.train_batch = None
        self.val_batch = None
        self.test_batch = None

        self.train_regions = []
        self.val_regions = []
        self.test_regions = []

        self.train_group_ids = []
        self.val_group_ids = []
        self.test_group_ids = []

        self.vertebra_groups = {}

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.model = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None

        self.history = []
        self.resumed_from_checkpoint = False
        self.current_fold = 1
        self.total_folds = 1

        self.experiment_start_time = None
        self.model_start_time = None
        self.fold_start_time = None
        self.fold_end_time = None
        self.experiment_end_time = None
        self.model_end_time = None

        set_seed(self.seed)

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _save_figure(self, fig, path):
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _metric_for_best(self, metrics_dict):
        val = metrics_dict.get(self.best_metric_name, None)
        if val is None:
            return -1e18
        if self.best_metric_name.lower() == "hausdorff":
            return -float(val)
        return float(val)

    def _resolve_path(self, rel_or_abs):
        p = str(rel_or_abs)
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    # ---------------------------------------------------------
    # DISCORD (delegado a DiscordWebhookNotifier)
    # ---------------------------------------------------------
    def _discord_notify_epoch(self, epoch, train_result, val_metrics, is_best, epoch_image_paths):
        if self.notifier:
            self.notifier.send_unet_epoch(
                epoch=epoch,
                total_epochs=self.epochs,
                train_result=train_result,
                val_metrics=val_metrics,
                is_best=is_best,
                best_metric_name=self.best_metric_name,
                file_paths=epoch_image_paths,
            )

    def _discord_notify_complete(self, summary, final_image_paths):
        if self.notifier:
            self.notifier.send_unet_complete(
                summary=summary,
                best_metric_name=self.best_metric_name,
                file_paths=final_image_paths,
            )

    # ---------------------------------------------------------
    # DATA — extracción de regiones
    # ---------------------------------------------------------
    def load_data(self):
        """Lee el CSV, split train/val/test, extrae regiones y agrupa."""
        self.df = pd.read_csv(self.index_csv).copy()

        if "split" in self.df.columns:
            self.df["_split"] = self.df["split"].astype(str).map(normalize_split_value)
            known = {"train", "val", "test"}

            if set(self.df["_split"].unique()) & known:
                self.train_df = self.df[self.df["_split"] == "train"].copy()
                self.val_df = self.df[self.df["_split"] == "val"].copy()
                self.test_df = self.df[self.df["_split"] == "test"].copy()

                if len(self.train_df) == 0 or len(self.val_df) == 0:
                    print("Split incompleto. Se hará split automático.")
                    self._auto_split()
                else:
                    print("Distribución split detectada:")
                    print(self.df["_split"].value_counts(dropna=False))
            else:
                print("Columna split no contiene train/val/test. Split automático.")
                self._auto_split()
        else:
            print("No existe columna split. Split automático.")
            self._auto_split()

        print(f"\nImágenes — train: {len(self.train_df)}  val: {len(self.val_df)}  test: {len(self.test_df)}")

        # Extraer regiones vertebrales para cada split
        print("\nExtrayendo regiones vertebrales ...")
        self.train_batch = self._extract_batch(self.train_df, "train")
        self.val_batch = self._extract_batch(self.val_df, "val")
        self.test_batch = self._extract_batch(self.test_df, "test")

        self.train_regions, self.train_group_ids = self._collect_regions_and_groups(self.train_batch)
        self.val_regions, self.val_group_ids = self._collect_regions_and_groups(self.val_batch)
        self.test_regions, self.test_group_ids = self._collect_regions_and_groups(self.test_batch)

        self._build_group_summary()

        print(f"\nParches — train: {len(self.train_regions)}  val: {len(self.val_regions)}  test: {len(self.test_regions)}")
        print(f"Grupos por nº vértebras: {sorted(self.vertebra_groups.keys())}")

    def _auto_split(self):
        df = self.df.copy()
        stratify_col = None
        for col in ["class_name", "label", "diagnosis", "target", "split"]:
            if col in df.columns:
                stratify_col = df[col]
                break

        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=self.seed, shuffle=True,
            stratify=stratify_col,
        )

        temp_stratify = temp_df[stratify_col.name] if stratify_col is not None else None
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=self.seed, shuffle=True,
            stratify=temp_stratify,
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

    def _extract_batch(self, df, split_name=""):
        image_paths = df[self.image_col].apply(self._resolve_path).tolist()
        mask_paths = df[self.mask_col].apply(self._resolve_path).tolist()

        batch = VertebraRegionBatch(
            image_paths=image_paths,
            mask_paths=mask_paths,
            **self.extractor_params,
        )
        batch.process()

        n_fail = len(batch.failed_images)
        print(f"  [{split_name}] imágenes={batch.total_images_processed}  "
              f"regiones={batch.total_regions}  fallidas={n_fail}")
        if n_fail > 0:
            for name, reason in batch.failed_images:
                print(f"      FAIL {name}: {reason}")

        return batch

    @staticmethod
    def _collect_regions_and_groups(batch):
        """Devuelve (regions_flat, group_ids) donde group_id = nº vértebras de la imagen."""
        regions = []
        group_ids = []

        for name in batch.image_names:
            proxy = batch.by_image.get(name)
            if proxy is None or len(proxy) == 0:
                continue
            count = len(proxy)
            for r in proxy.regions:
                regions.append(r)
                group_ids.append(count)

        return regions, group_ids

    def _build_group_summary(self):
        """Construye resumen de grupos por número de vértebras."""
        groups = defaultdict(lambda: {"train_images": 0, "val_images": 0, "test_images": 0,
                                       "train_regions": 0, "val_regions": 0, "test_regions": 0})

        for split_name, batch in [("train", self.train_batch),
                                   ("val", self.val_batch),
                                   ("test", self.test_batch)]:
            for name, proxy in batch.by_image.items():
                count = len(proxy)
                groups[count][f"{split_name}_images"] += 1
                groups[count][f"{split_name}_regions"] += count

        self.vertebra_groups = dict(sorted(groups.items()))

    # ---------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------
    def build_loaders(self):
        train_ds = VertebraRegionDataset(self.train_regions, self.patch_size, self.binarize_mask)
        val_ds = VertebraRegionDataset(self.val_regions, self.patch_size, self.binarize_mask)
        test_ds = VertebraRegionDataset(self.test_regions, self.patch_size, self.binarize_mask)

        pin = torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=pin,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=pin,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=pin,
        )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    def build_model(self):
        if self.model_type == "baseline":
            self.model = UNetBaseline(1, 1, self.base_channels)
        elif self.model_type == "variance_input":
            self.model = UNetVarianceInput(1, 1, self.base_channels)
        elif self.model_type == "variance_branch":
            self.model = UNetVarianceBranch(1, 1, self.base_channels)
        else:
            raise ValueError(f"model.type no soportado: {self.model_type}")

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.resume_checkpoint_path and os.path.exists(self.resume_checkpoint_path):
            print(f"Reanudando desde checkpoint: {self.resume_checkpoint_path}")
            state = torch.load(self.resume_checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
                if "optimizer_state_dict" in state:
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.resumed_from_checkpoint = True

    # ---------------------------------------------------------
    # TRAIN / EVAL
    # ---------------------------------------------------------
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_steps = 0
        t0 = time.time()

        for imgs, masks in self.train_loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_samples += imgs.size(0)
            total_steps += 1

        duration = time.time() - t0
        return {
            "train_loss": total_loss / max(len(self.train_loader), 1),
            "samples_per_sec": total_samples / max(duration, 1e-8),
            "steps_per_sec": total_steps / max(duration, 1e-8),
            "train_duration_sec": duration,
        }

    @torch.no_grad()
    def evaluate(self, loader, split_name="val"):
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        hd_list = []
        num_batches = 0

        for imgs, masks in loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            logits = self.model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            loss = self.criterion(logits, masks)

            dice = dice_from_probs(preds, masks)
            iou = iou_from_probs(preds, masks)
            prec = precision_from_probs(preds, masks)
            rec = recall_from_probs(preds, masks)
            f1 = f1_from_precision_recall(prec, rec)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_precision += prec
            total_recall += rec
            total_f1 += f1
            num_batches += 1

            preds_np = preds.cpu().numpy()
            tgts_np = masks.cpu().numpy()
            for p, t in zip(preds_np, tgts_np):
                hd = hausdorff_distance_binary(p[0], t[0])
                if not np.isnan(hd):
                    hd_list.append(hd)

        nb = max(num_batches, 1)
        return {
            f"{split_name}_loss": total_loss / nb,
            "dice": total_dice / nb,
            "iou": total_iou / nb,
            "precision": total_precision / nb,
            "recall": total_recall / nb,
            "f1": total_f1 / nb,
            "hausdorff": float(np.mean(hd_list)) if hd_list else np.nan,
        }

    @torch.no_grad()
    def evaluate_by_group(self, loader, group_ids, split_name="test"):
        """Evalúa el modelo y agrega métricas por grupo de vértebras."""
        self.model.eval()
        group_metrics = defaultdict(lambda: {
            "dice": [], "iou": [], "precision": [], "recall": [], "f1": [],
        })

        sample_idx = 0
        for imgs, masks in loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            logits = self.model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            bs = imgs.size(0)
            for i in range(bs):
                p = preds[i: i + 1]
                m = masks[i: i + 1]

                gid = group_ids[sample_idx]
                group_metrics[gid]["dice"].append(dice_from_probs(p, m))
                group_metrics[gid]["iou"].append(iou_from_probs(p, m))
                prec = precision_from_probs(p, m)
                rec = recall_from_probs(p, m)
                group_metrics[gid]["precision"].append(prec)
                group_metrics[gid]["recall"].append(rec)
                group_metrics[gid]["f1"].append(f1_from_precision_recall(prec, rec))
                sample_idx += 1

        result = {}
        for gid in sorted(group_metrics):
            gm = group_metrics[gid]
            result[str(gid)] = {
                "n_vertebras": gid,
                "n_parches": len(gm["dice"]),
                "dice": float(np.mean(gm["dice"])),
                "iou": float(np.mean(gm["iou"])),
                "precision": float(np.mean(gm["precision"])),
                "recall": float(np.mean(gm["recall"])),
                "f1": float(np.mean(gm["f1"])),
            }
        return result

    # ---------------------------------------------------------
    # GUARDADO
    # ---------------------------------------------------------
    def save_config(self):
        save_json(os.path.join(self.output_dir, "config.json"), self.config)

    def save_history(self):
        pd.DataFrame(self.history).to_csv(
            os.path.join(self.output_dir, "history.csv"), index=False)

    def save_summary(self, summary):
        save_json(os.path.join(self.output_dir, "summary.json"), summary)

    def save_checkpoint(self, filename="best_model.pt", epoch=None, metrics=None):
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": utc_now_iso(),
            "config": self.config,
        }, path)

    # ---------------------------------------------------------
    # VISUALIZACIONES DE GRUPOS
    # ---------------------------------------------------------
    def save_group_distribution(self):
        """Gráfico de barras: nº de imágenes por grupo de vértebras, por split."""
        counts = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

        for split_name, batch in [("train", self.train_batch),
                                   ("val", self.val_batch),
                                   ("test", self.test_batch)]:
            for name, proxy in batch.by_image.items():
                counts[split_name][len(proxy)] += 1

        all_groups = sorted(set().union(*[c.keys() for c in counts.values()]))
        x = np.arange(len(all_groups))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(all_groups) * 1.2), 5))

        for i, (split_name, color) in enumerate([("train", "#4c72b0"),
                                                   ("val", "#dd8452"),
                                                   ("test", "#55a868")]):
            vals = [counts[split_name].get(g, 0) for g in all_groups]
            bars = ax.bar(x + i * width, vals, width, label=split_name, color=color)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            str(v), ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Nº de vértebras detectadas")
        ax.set_ylabel("Nº de imágenes")
        ax.set_title("Distribución de imágenes por grupo de vértebras")
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_groups)
        ax.legend()
        plt.tight_layout()
        self._save_figure(fig, os.path.join(self.groups_dir, "group_distribution.png"))

    def save_group_patch_samples(self, max_groups=8, patches_per_group=4):
        """Muestra parches de muestra por cada grupo de vértebras."""
        group_regions = defaultdict(list)
        for r, gid in zip(self.train_regions + self.val_regions, self.train_group_ids + self.val_group_ids):
            group_regions[gid].append(r)

        groups = sorted(group_regions.keys())[:max_groups]
        if not groups:
            return

        n_cols = patches_per_group * 2
        n_rows = len(groups)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
        if n_rows == 1:
            axes = [axes]

        for row, gid in enumerate(groups):
            regions = group_regions[gid]
            samples = regions[:patches_per_group]

            for col_pair in range(patches_per_group):
                col_img = col_pair * 2
                col_mask = col_pair * 2 + 1
                ax_img = axes[row][col_img]
                ax_mask = axes[row][col_mask]

                if col_pair < len(samples):
                    r = samples[col_pair]
                    ax_img.imshow(r.patch_img, cmap="gray")
                    ax_mask.imshow(r.patch_mask, cmap="gray")
                else:
                    ax_img.axis("off")
                    ax_mask.axis("off")
                    continue

                ax_img.set_title(f"g={gid}" if col_pair == 0 else "", fontsize=9)
                ax_img.axis("off")
                ax_mask.axis("off")

            axes[row][0].set_ylabel(f"{gid} vért.", fontsize=9, rotation=0, labelpad=40, va="center")

        plt.suptitle("Parches de muestra por grupo de vértebras", fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, os.path.join(self.groups_dir, "group_patch_samples.png"))

    def save_group_metrics_plot(self, group_metrics):
        """Gráfico de métricas por grupo de vértebras."""
        if not group_metrics:
            return

        groups = sorted(group_metrics.keys(), key=lambda k: int(k))
        gids = [int(g) for g in groups]
        dice_vals = [group_metrics[g]["dice"] for g in groups]
        iou_vals = [group_metrics[g]["iou"] for g in groups]
        f1_vals = [group_metrics[g]["f1"] for g in groups]
        n_patches = [group_metrics[g]["n_parches"] for g in groups]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(gids))
        w = 0.25
        ax1.bar(x - w, dice_vals, w, label="Dice", color="#4c72b0")
        ax1.bar(x, iou_vals, w, label="IoU", color="#dd8452")
        ax1.bar(x + w, f1_vals, w, label="F1", color="#55a868")
        ax1.set_xlabel("Nº de vértebras")
        ax1.set_ylabel("Métrica")
        ax1.set_title("Métricas por grupo de vértebras (test)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(gids)
        ax1.legend()
        ax1.set_ylim(0, 1.05)

        ax2.bar(x, n_patches, color="#8c564b")
        ax2.set_xlabel("Nº de vértebras")
        ax2.set_ylabel("Nº parches")
        ax2.set_title("Cantidad de parches por grupo")
        ax2.set_xticks(x)
        ax2.set_xticklabels(gids)
        for xi, v in zip(x, n_patches):
            ax2.text(xi, v + 0.5, str(v), ha="center", fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, os.path.join(self.groups_dir, "group_test_metrics.png"))

    # ---------------------------------------------------------
    # VISUALIZACIONES DE ENTRENAMIENTO
    # ---------------------------------------------------------
    @torch.no_grad()
    def save_variance_filters(self, split="val", max_samples=3, epoch=None):
        loader = {"train": self.train_loader, "test": self.test_loader}.get(split, self.val_loader)

        imgs, _ = next(iter(loader))
        imgs = imgs.to(self.device)

        var_layer = VarianceInputLayer(kernel_sizes=(3, 5, 9)).to(self.device)
        stacked = var_layer(imgs).cpu().numpy()

        n = min(max_samples, stacked.shape[0])
        for i in range(n):
            original = stacked[i, 0]
            v3, v5, v9 = stacked[i, 1], stacked[i, 2], stacked[i, 3]

            fig = plt.figure(figsize=(16, 8))

            plt.subplot(2, 4, 1); plt.imshow(original, cmap="gray"); plt.title("Parche"); plt.axis("off")
            plt.subplot(2, 4, 2); plt.imshow(v3, cmap="hot"); plt.title("Var k=3"); plt.axis("off")
            plt.subplot(2, 4, 3); plt.imshow(v5, cmap="hot"); plt.title("Var k=5"); plt.axis("off")
            plt.subplot(2, 4, 4); plt.imshow(v9, cmap="hot"); plt.title("Var k=9"); plt.axis("off")

            plt.subplot(2, 4, 5); plt.imshow(original, cmap="gray"); plt.title("Parche"); plt.axis("off")
            plt.subplot(2, 4, 6); plt.imshow(original, cmap="gray"); plt.imshow(v3, cmap="jet", alpha=0.35); plt.title("Overlay k=3"); plt.axis("off")
            plt.subplot(2, 4, 7); plt.imshow(original, cmap="gray"); plt.imshow(v5, cmap="jet", alpha=0.35); plt.title("Overlay k=5"); plt.axis("off")
            plt.subplot(2, 4, 8); plt.imshow(original, cmap="gray"); plt.imshow(v9, cmap="jet", alpha=0.35); plt.title("Overlay k=9"); plt.axis("off")

            plt.tight_layout()
            fname = f"{split}_sample_{i:03d}.png" if epoch is None else f"epoch_{epoch:03d}_{split}_sample_{i:03d}.png"
            self._save_figure(fig, os.path.join(self.filters_dir, fname))

    @torch.no_grad()
    def save_predictions_grid(self, split="val", n=3, epoch=None, out_dir=None):
        loader = {"train": self.train_loader, "test": self.test_loader}.get(split, self.val_loader)

        out_dir = out_dir or self.epoch_samples_dir
        ensure_dir(out_dir)

        self.model.eval()
        imgs, masks = next(iter(loader))
        logits = self.model(imgs.to(self.device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(np.float32)

        imgs_np = imgs.numpy()
        masks_np = masks.numpy()
        n = min(n, len(imgs_np))

        fig = plt.figure(figsize=(18, 4 * n))
        for i in range(n):
            plt.subplot(n, 5, 5 * i + 1); plt.imshow(imgs_np[i, 0], cmap="gray"); plt.title(f"{split} | parche"); plt.axis("off")
            plt.subplot(n, 5, 5 * i + 2); plt.imshow(masks_np[i, 0], cmap="gray"); plt.title("GT"); plt.axis("off")
            plt.subplot(n, 5, 5 * i + 3); plt.imshow(probs[i, 0], cmap="viridis"); plt.title("Prob"); plt.axis("off")
            plt.subplot(n, 5, 5 * i + 4); plt.imshow(preds[i, 0], cmap="gray"); plt.title("Pred"); plt.axis("off")
            plt.subplot(n, 5, 5 * i + 5); plt.imshow(imgs_np[i, 0], cmap="gray"); plt.imshow(preds[i, 0], cmap="jet", alpha=0.35); plt.title("Overlay"); plt.axis("off")

        plt.tight_layout()
        fname = f"{split}_predictions.png" if epoch is None else f"epoch_{epoch:03d}_{split}_predictions.png"
        self._save_figure(fig, os.path.join(out_dir, fname))

    def save_history_plot(self, epoch=None):
        if not self.history:
            return
        df = pd.DataFrame(self.history)

        fig = plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.legend(); plt.title("Loss")

        plt.subplot(1, 3, 2)
        plt.plot(df["epoch"], df["val_dice"], label="val_dice")
        plt.plot(df["epoch"], df["val_iou"], label="val_iou")
        plt.legend(); plt.title("Dice / IoU")

        plt.subplot(1, 3, 3)
        plt.plot(df["epoch"], df["val_hausdorff"], label="val_hausdorff")
        plt.legend(); plt.title("Hausdorff")

        plt.tight_layout()
        fname = "history_curves.png" if epoch is None else f"epoch_{epoch:03d}_curves.png"
        self._save_figure(fig, os.path.join(self.plots_dir, fname))

    # ---------------------------------------------------------
    # EPOCH STATUS
    # ---------------------------------------------------------
    def build_epoch_status(
        self, epoch, total_epochs, train_result, val_metrics,
        best_epoch, best_metric_value,
        epoch_start_time, epoch_end_time,
        avg_epoch_duration_sec, estimated_remaining_sec, checkpoint_saved_at
    ):
        sysm = get_system_metrics()

        epoch_duration_sec = (
            datetime.fromisoformat(epoch_end_time.replace("Z", "+00:00"))
            - datetime.fromisoformat(epoch_start_time.replace("Z", "+00:00"))
        ).total_seconds()

        status = {
            "status": "running" if epoch < total_epochs else "completed",
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "resume_checkpoint_path": self.resume_checkpoint_path,
            "current_fold": self.current_fold,
            "total_folds": self.total_folds,
            "current_model": self.model_name,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "train_size": len(self.train_regions),
            "val_size": len(self.val_regions),
            "test_size": len(self.test_regions),
            "train_images": len(self.train_df),
            "val_images": len(self.val_df),
            "test_images": len(self.test_df),
            "experiment_start_time": self.experiment_start_time,
            "model_start_time": self.model_start_time,
            "total_epochs": total_epochs,
            "current_epoch": epoch,
            "epoch_start_time": epoch_start_time,
            "fold_start_time": self.fold_start_time,
            "fold_end_time": self.fold_end_time,
            "fold_duration_sec": (
                datetime.fromisoformat(self.fold_end_time.replace("Z", "+00:00"))
                - datetime.fromisoformat(self.fold_start_time.replace("Z", "+00:00"))
            ).total_seconds() if self.fold_end_time else None,
            "best_epoch": best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": best_metric_value,
            "epoch_end_time": epoch_end_time,
            "epoch_duration_sec": epoch_duration_sec,
            "avg_epoch_duration_sec": avg_epoch_duration_sec,
            "estimated_remaining_sec": estimated_remaining_sec,
            "last_completed_step": "epoch_end",
            "checkpoint_saved_at": checkpoint_saved_at,
            "timestamp": utc_now_iso(),
            "model_name": self.model_name,
            "fold": self.current_fold,
            "epoch": epoch,
            "samples_per_sec": train_result["samples_per_sec"],
            "steps_per_sec": train_result["steps_per_sec"],
            "val_metrics": val_metrics,
            "vertebra_groups": {str(k): v for k, v in self.vertebra_groups.items()},
        }
        status.update(sysm)
        return status

    # ---------------------------------------------------------
    # FIT
    # ---------------------------------------------------------
    def fit(self):
        self.experiment_start_time = utc_now_iso()
        self.model_start_time = utc_now_iso()
        self.fold_start_time = utc_now_iso()

        self.load_data()
        self.build_loaders()
        self.build_model()
        self.save_config()

        # Plots de grupos (antes de entrenar)
        self.save_group_distribution()
        self.save_group_patch_samples()

        best_score = -1e18
        best_val_metrics = None
        best_epoch = None
        epoch_durations = []

        epoch_jsonl_path = os.path.join(self.output_dir, "epoch_metrics.jsonl")

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = utc_now_iso()
            t_epoch = time.time()

            train_result = self.train_one_epoch()
            val_metrics = self.evaluate(self.val_loader, split_name="val")

            current_score = self._metric_for_best(val_metrics)
            is_best = current_score > best_score

            if is_best:
                best_score = current_score
                best_val_metrics = val_metrics
                best_epoch = epoch
                self.save_checkpoint("best_model.pt", epoch=epoch, metrics=val_metrics)

            self.save_checkpoint("last_model.pt", epoch=epoch, metrics=val_metrics)

            epoch_duration = time.time() - t_epoch
            epoch_durations.append(epoch_duration)
            avg_epoch_duration_sec = float(np.mean(epoch_durations))
            estimated_remaining_sec = avg_epoch_duration_sec * (self.epochs - epoch)

            epoch_end_time = utc_now_iso()
            checkpoint_saved_at = utc_now_iso()

            row = {
                "epoch": epoch,
                "train_loss": train_result["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_hausdorff": val_metrics["hausdorff"],
                "samples_per_sec": train_result["samples_per_sec"],
                "steps_per_sec": train_result["steps_per_sec"],
                "epoch_duration_sec": epoch_duration,
                "is_best": is_best,
            }
            self.history.append(row)
            self.save_history()

            self.save_history_plot(epoch=epoch)
            self.save_history_plot(epoch=None)

            if self.save_visuals_each_epoch:
                self.save_predictions_grid(split="train", n=self.num_visual_samples, epoch=epoch)
                self.save_predictions_grid(split="val", n=self.num_visual_samples, epoch=epoch)
                self.save_variance_filters(split="val", max_samples=self.num_visual_samples, epoch=epoch)

            status_dict = self.build_epoch_status(
                epoch=epoch, total_epochs=self.epochs,
                train_result=train_result, val_metrics=val_metrics,
                best_epoch=best_epoch,
                best_metric_value=(best_val_metrics[self.best_metric_name]
                                   if best_val_metrics and self.best_metric_name in best_val_metrics else None),
                epoch_start_time=epoch_start_time,
                epoch_end_time=epoch_end_time,
                avg_epoch_duration_sec=avg_epoch_duration_sec,
                estimated_remaining_sec=estimated_remaining_sec,
                checkpoint_saved_at=checkpoint_saved_at,
            )
            append_jsonl(epoch_jsonl_path, status_dict)

            print(
                f"[Epoch {epoch}/{self.epochs}] "
                f"train_loss={train_result['train_loss']:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"dice={val_metrics['dice']:.4f} "
                f"iou={val_metrics['iou']:.4f} "
                f"precision={val_metrics['precision']:.4f} "
                f"recall={val_metrics['recall']:.4f} "
                f"f1={val_metrics['f1']:.4f} "
                f"hausdorff={val_metrics['hausdorff']:.4f}"
            )

            # --- Discord: notificar cada N épocas ---
            if self.notifier and epoch % self.discord_notify_every == 0:
                epoch_images = []
                curves_path = os.path.join(self.plots_dir, f"epoch_{epoch:03d}_curves.png")
                if os.path.isfile(curves_path):
                    epoch_images.append(curves_path)
                if self.save_visuals_each_epoch:
                    val_pred_path = os.path.join(self.epoch_samples_dir, f"epoch_{epoch:03d}_val_predictions.png")
                    if os.path.isfile(val_pred_path):
                        epoch_images.append(val_pred_path)
                self._discord_notify_epoch(epoch, train_result, val_metrics, is_best, epoch_images)

        # ----- EVALUACIÓN FINAL -----
        self.fold_end_time = utc_now_iso()

        best_model_path = os.path.join(self.checkpoints_dir, "best_model.pt")
        best_payload = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(best_payload["model_state_dict"])

        test_metrics = self.evaluate(self.test_loader, split_name="test")
        test_group_metrics = self.evaluate_by_group(self.test_loader, self.test_group_ids, "test")

        self.save_predictions_grid(split="val", n=self.num_visual_samples, out_dir=self.best_predictions_dir)
        self.save_predictions_grid(split="test", n=self.num_visual_samples, out_dir=self.best_predictions_dir)
        self.save_variance_filters(split="test", max_samples=self.num_visual_samples)
        self.save_group_metrics_plot(test_group_metrics)

        save_json(os.path.join(self.groups_dir, "test_group_metrics.json"), test_group_metrics)

        self.model_end_time = utc_now_iso()
        self.experiment_end_time = utc_now_iso()

        experiment_duration_sec = (
            datetime.fromisoformat(self.experiment_end_time.replace("Z", "+00:00"))
            - datetime.fromisoformat(self.experiment_start_time.replace("Z", "+00:00"))
        ).total_seconds()
        model_duration_sec = (
            datetime.fromisoformat(self.model_end_time.replace("Z", "+00:00"))
            - datetime.fromisoformat(self.model_start_time.replace("Z", "+00:00"))
        ).total_seconds()

        final_system_metrics = get_system_metrics()

        summary = {
            "status": "completed",
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "resume_checkpoint_path": self.resume_checkpoint_path,
            "current_fold": self.current_fold,
            "total_folds": self.total_folds,
            "current_model": self.model_name,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "train_images": len(self.train_df),
            "val_images": len(self.val_df),
            "test_images": len(self.test_df),
            "train_patches": len(self.train_regions),
            "val_patches": len(self.val_regions),
            "test_patches": len(self.test_regions),
            "patch_size": list(self.patch_size),
            "experiment_start_time": self.experiment_start_time,
            "model_start_time": self.model_start_time,
            "total_epochs": self.epochs,
            "current_epoch": self.epochs,
            "fold_start_time": self.fold_start_time,
            "fold_end_time": self.fold_end_time,
            "fold_duration_sec": (
                datetime.fromisoformat(self.fold_end_time.replace("Z", "+00:00"))
                - datetime.fromisoformat(self.fold_start_time.replace("Z", "+00:00"))
            ).total_seconds(),
            "best_epoch": best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": (best_val_metrics[self.best_metric_name]
                                   if best_val_metrics and self.best_metric_name in best_val_metrics else None),
            "epoch_end_time": utc_now_iso(),
            "epoch_duration_sec": self.history[-1]["epoch_duration_sec"] if self.history else None,
            "avg_epoch_duration_sec": float(np.mean([h["epoch_duration_sec"] for h in self.history])) if self.history else None,
            "estimated_remaining_sec": 0.0,
            "last_completed_step": "epoch_end",
            "checkpoint_saved_at": utc_now_iso(),
            "timestamp": utc_now_iso(),
            "model_name": self.model_name,
            "fold": self.current_fold,
            "epoch": self.epochs,
            "samples_per_sec": self.history[-1]["samples_per_sec"] if self.history else None,
            "steps_per_sec": self.history[-1]["steps_per_sec"] if self.history else None,
            "test_metrics": test_metrics,
            "test_metrics_by_group": test_group_metrics,
            "vertebra_groups": {str(k): v for k, v in self.vertebra_groups.items()},
            "model_end_time": self.model_end_time,
            "experiment_end_time": self.experiment_end_time,
            "experiment_duration_sec": experiment_duration_sec,
            "model_duration_sec": model_duration_sec,
            "execution_mode": self.execution_mode,
            "model_type": self.model_type,
            "device": self.device,
            "output_dir": self.output_dir,
        }
        summary.update(final_system_metrics)

        self.save_summary(summary)

        # --- Discord: notificar fin ---
        if self.notifier:
            final_images = [
                os.path.join(self.plots_dir, "history_curves.png"),
                os.path.join(self.best_predictions_dir, "test_predictions.png"),
                os.path.join(self.groups_dir, "group_test_metrics.png"),
            ]
            final_images = [p for p in final_images if os.path.isfile(p)]
            self._discord_notify_complete(summary, final_images)

        return summary

    # ---------------------------------------------------------
    # MÉTODOS DE APOYO INTERACTIVO
    # ---------------------------------------------------------
    @torch.no_grad()
    def show_predictions(self, split="val", n=3):
        loader = {"train": self.train_loader, "test": self.test_loader}.get(split, self.val_loader)

        self.model.eval()
        imgs, masks = next(iter(loader))
        logits = self.model(imgs.to(self.device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(np.float32)

        imgs_np = imgs.numpy()
        masks_np = masks.numpy()
        n = min(n, len(imgs_np))

        plt.figure(figsize=(16, 4 * n))
        for i in range(n):
            plt.subplot(n, 4, 4 * i + 1); plt.imshow(imgs_np[i, 0], cmap="gray"); plt.title(f"{split} | parche"); plt.axis("off")
            plt.subplot(n, 4, 4 * i + 2); plt.imshow(masks_np[i, 0], cmap="gray"); plt.title("GT"); plt.axis("off")
            plt.subplot(n, 4, 4 * i + 3); plt.imshow(probs[i, 0], cmap="viridis"); plt.title("Prob"); plt.axis("off")
            plt.subplot(n, 4, 4 * i + 4); plt.imshow(preds[i, 0], cmap="gray"); plt.title("Pred"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_history(self):
        if not self.history:
            print("No hay history todavía.")
            return
        df = pd.DataFrame(self.history)
        print(df.tail())

        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.legend(); plt.title("Loss")

        plt.subplot(1, 3, 2)
        plt.plot(df["epoch"], df["val_dice"], label="val_dice")
        plt.plot(df["epoch"], df["val_iou"], label="val_iou")
        plt.plot(df["epoch"], df["val_f1"], label="val_f1")
        plt.legend(); plt.title("Dice / IoU / F1")

        plt.subplot(1, 3, 3)
        plt.plot(df["epoch"], df["val_hausdorff"], label="val_hausdorff")
        plt.legend(); plt.title("Hausdorff")

        plt.tight_layout()
        plt.show()
