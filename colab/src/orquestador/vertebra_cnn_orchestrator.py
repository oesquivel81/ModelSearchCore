# src/orquestador/vertebra_cnn_orchestrator.py

import os
import time
import traceback
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.helpers import (
    ensure_dir,
    save_json,
    append_jsonl,
    set_seed,
    utc_now_iso,
    get_system_metrics,
)
from src.utils.metrics import classification_metrics_from_logits
from src.utils.discord_webhook_notifier import DiscordWebhookNotifier

from src.extractor.vertebra_region_extractor import (
    build_study_split,
    save_study_split,
    VertebraRegionExtractor,
)
from src.extractor.vertebra_subpatch_generator import VertebraSubpatchGenerator

from src.varianza_patches_cnn.models import (
    BaselinePatchCNN,
    VarianceInputPatchCNN,
    VarianceBranchPatchCNN,
)
from src.varianza_patches_cnn.vertebra_patch_dataset_v2 import (
    VertebraSubpatchDatasetFlexible,
)
from src.utils.variance_patch_builder import VariancePatchBuilderV2


class VertebraCNNOrchestrator:
    """
    Orquestador completo:
      split -> regiones vertebrales -> subpatches -> builder varianza -> dataset -> CNN -> métricas -> Discord
    """

    def _log(self, msg):
        elapsed = time.time() - self._t0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print(f"[{h:02d}:{m:02d}:{s:02d}] [{self.experiment_name}] {msg}", flush=True)

    def __init__(self, cfg: dict):
        self._t0 = time.time()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.experiment_name = cfg["experiment_name"]
        self.output_dir = ensure_dir(os.path.join(cfg["drive_root"], self.experiment_name))

        self.split_csv = os.path.join(self.output_dir, "study_split.csv")
        self.vertebra_root = ensure_dir(os.path.join(self.output_dir, "vertebra_regions"))
        self.subpatch_root = ensure_dir(os.path.join(self.output_dir, "vertebra_subpatches"))
        self.processed_root = ensure_dir(os.path.join(self.output_dir, "processed"))

        self.vertebra_csv = os.path.join(self.vertebra_root, "vertebra_regions_metadata.csv")
        self.subpatch_csv = os.path.join(self.subpatch_root, "vertebra_subpatches_metadata.csv")

        self.history_csv = os.path.join(self.output_dir, "history.csv")
        self.history_jsonl = os.path.join(self.output_dir, "epoch_metrics.jsonl")
        self.best_model_path = os.path.join(self.output_dir, "best_model.pt")
        self.result_json = os.path.join(self.output_dir, "result.json")

        set_seed(cfg.get("seed", 42))
        save_json(os.path.join(self.output_dir, "config.json"), cfg)

        discord_cfg = cfg.get("discord", {})
        self.notifier = DiscordWebhookNotifier(
            webhook_url=discord_cfg.get("webhook_url"),
            experiment_name=self.experiment_name
        )
        self._log(f"Inicializado | device={self.device} | output={self.output_dir}")

    # =========================================================
    # PREPARACIÓN
    # =========================================================
    def run_split(self):
        self._log("Split: generando partición train/val/test...")
        split_cfg = self.cfg.get("split", {})
        extractor_cfg = self.cfg["extractor"]

        split_df = build_study_split(
            index_csv=extractor_cfg["index_csv"],
            image_col=extractor_cfg["image_col"],
            seed=self.cfg.get("seed", 42),
            train_size=split_cfg.get("train_size", 0.70),
            val_size=split_cfg.get("val_size", 0.15),
            test_size=split_cfg.get("test_size", 0.15),
        )
        save_study_split(split_df, self.split_csv)
        counts = split_df["split"].value_counts().to_dict()
        self._log(f"Split completado: {counts}")
        return split_df

    def run_vertebra_extraction(self):
        self._log("Extracción: extrayendo regiones vertebrales...")
        extractor_cfg = self.cfg["extractor"]

        extractor = VertebraRegionExtractor(
            base_dir=extractor_cfg["base_dir"],
            image_col=extractor_cfg["image_col"],
            mask_col=extractor_cfg["mask_col"],
            min_area=extractor_cfg.get("min_area", 50),
            pad_x=extractor_cfg.get("pad_x", 30),
            pad_y=extractor_cfg.get("pad_y", 15),
            save_root=self.vertebra_root,
        )

        vertebra_df = extractor.extract_all(
            index_csv=extractor_cfg["index_csv"],
            split_csv=self.split_csv,
        )
        self._log(f"Extracción completada: {len(vertebra_df)} regiones vertebrales")
        return vertebra_df

    def run_subpatch_generation(self):
        self._log("Subpatches: generando parches...")
        data_cfg = self.cfg["data"]
        sub_cfg = self.cfg.get("subpatches", {})

        subpatcher = VertebraSubpatchGenerator(
            patch_size=tuple(data_cfg["patch_size"]),
            subpatch_size=tuple(sub_cfg.get("subpatch_size", [32, 32])),
            stride=tuple(sub_cfg.get("stride", [32, 32])),
            save_root=self.subpatch_root,
        )

        subpatch_df = subpatcher.generate_all(self.vertebra_csv)
        self._log(f"Subpatches completados: {len(subpatch_df)} subpatches")
        return subpatch_df

    def prepare(self):
        self._log("═" * 50)
        self._log("FASE 1: PREPARACIÓN DE DATOS")
        self._log("═" * 50)
        split_df = self.run_split()
        vertebra_df = self.run_vertebra_extraction()
        subpatch_df = self.run_subpatch_generation()

        summary = {
            "num_studies": int(split_df["study_id"].nunique()),
            "num_vertebra_regions": int(len(vertebra_df)),
            "num_subpatches": int(len(subpatch_df)),
            "device": self.device,
        }

        self.notifier.send_text(
            f"🚀 Preparación completada\n"
            f"studies={summary['num_studies']} | "
            f"vertebras={summary['num_vertebra_regions']} | "
            f"subpatches={summary['num_subpatches']} | "
            f"device={summary['device']}"
        )
        return summary

    # =========================================================
    # BUILDER / DATASETS / DATALOADERS
    # =========================================================
    def build_variance_builder(self):
        data_cfg = self.cfg["data"]
        variance_cfg = self.cfg.get("variance", {})
        sub_cfg = self.cfg.get("subpatches", {})

        return VariancePatchBuilderV2(
            patch_size=tuple(data_cfg["patch_size"]),
            variance_ksize=variance_cfg.get("variance_ksize", 5),
            save_root=self.processed_root,
            make_subpatches=False,
            subpatch_size=tuple(sub_cfg.get("subpatch_size", [32, 32])),
            subpatch_stride=tuple(sub_cfg.get("stride", [32, 32])),
        )

    def load_subpatch_metadata(self):
        return pd.read_csv(self.subpatch_csv)

    def build_datasets(self):
        self._log("Construyendo datasets (train/val/test)...")
        subpatch_df = self.load_subpatch_metadata()
        builder = self.build_variance_builder()

        model_type = self.cfg["model"]["type"]
        num_classes = self.cfg["model"].get("num_classes", 13)

        train_ds = VertebraSubpatchDatasetFlexible(
            metadata_df=subpatch_df,
            split="train",
            builder=builder,
            model_type=model_type,
            num_classes=num_classes,
        )
        val_ds = VertebraSubpatchDatasetFlexible(
            metadata_df=subpatch_df,
            split="val",
            builder=builder,
            model_type=model_type,
            num_classes=num_classes,
        )
        test_ds = VertebraSubpatchDatasetFlexible(
            metadata_df=subpatch_df,
            split="test",
            builder=builder,
            model_type=model_type,
            num_classes=num_classes,
        )
        self._log(f"Datasets: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
        return train_ds, val_ds, test_ds

    def build_dataloaders(self):
        train_ds, val_ds, test_ds = self.build_datasets()
        tr_cfg = self.cfg["training"]

        train_loader = DataLoader(
            train_ds,
            batch_size=tr_cfg.get("batch_size", 16),
            shuffle=True,
            num_workers=tr_cfg.get("num_workers", 2),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=tr_cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=tr_cfg.get("num_workers", 2),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=tr_cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=tr_cfg.get("num_workers", 2),
        )
        return train_loader, val_loader, test_loader

    # =========================================================
    # MODELO
    # =========================================================
    def build_model(self):
        model_cfg = self.cfg["model"]
        model_type = model_cfg["type"]
        base_channels = model_cfg.get("base_channels", 32)
        num_classes = model_cfg.get("num_classes", 13)

        if model_type == "baseline":
            model = BaselinePatchCNN(base_channels=base_channels, num_classes=num_classes)
        elif model_type == "variance_input":
            model = VarianceInputPatchCNN(base_channels=base_channels, num_classes=num_classes)
        elif model_type == "variance_branch":
            model = VarianceBranchPatchCNN(base_channels=base_channels, num_classes=num_classes)
        else:
            raise ValueError(f"model_type no soportado: {model_type}")

        model = model.to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        self._log(f"Modelo: {model_type} | params={n_params:,} | base_ch={base_channels} | classes={num_classes}")
        return model

    # =========================================================
    # FORWARD DE BATCH
    # =========================================================
    def _forward_batch(self, model, batch):
        model_type = self.cfg["model"]["type"]

        if model_type == "variance_branch":
            x_img = batch["image"].to(self.device)
            x_var = batch["variance"].to(self.device)
            y = batch["label"].to(self.device)
            logits = model(x_img, x_var)
            return logits, y

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = model(x)
        return logits, y

    # =========================================================
    # TRAIN / EVAL
    # =========================================================
    def train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()

        total_loss = 0.0
        total = 0
        all_logits = []
        all_targets = []

        for batch in loader:
            logits, y = self._forward_batch(model, batch)

            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        metrics = classification_metrics_from_logits(
            logits_cat,
            targets_cat,
            num_classes=self.cfg["model"].get("num_classes", 13)
        )
        metrics["loss"] = total_loss / max(1, total)
        return metrics

    def eval_one_epoch(self, model, loader, criterion):
        model.eval()

        total_loss = 0.0
        total = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                logits, y = self._forward_batch(model, batch)
                loss = criterion(logits, y)

                total_loss += loss.item() * y.size(0)
                total += y.size(0)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        metrics = classification_metrics_from_logits(
            logits_cat,
            targets_cat,
            num_classes=self.cfg["model"].get("num_classes", 13)
        )
        metrics["loss"] = total_loss / max(1, total)
        return metrics

    # =========================================================
    # FIT
    # =========================================================
    def fit(self):
        try:
            prep_summary = self.prepare()

            self._log("═" * 50)
            self._log("FASE 2: CONSTRUCCIÓN DEL MODELO")
            self._log("═" * 50)
            train_loader, val_loader, test_loader = self.build_dataloaders()
            model = self.build_model()

            tr_cfg = self.cfg["training"]
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg.get("lr", 1e-3))

            best_metric_name = tr_cfg.get("best_metric_name", "f1_macro")
            best_metric_value = -1.0
            best_epoch = -1
            history_rows = []

            self.notifier.send_text(
                f"📊 Inicio entrenamiento\n"
                f"device={self.device}\n"
                f"system={get_system_metrics()}"
            )

            total_epochs = tr_cfg.get("epochs", 20)

            self._log("═" * 50)
            self._log(f"FASE 3: ENTRENAMIENTO ({total_epochs} épocas)")
            self._log("═" * 50)

            for epoch in range(total_epochs):
                ep_start = time.time()
                tr_metrics = self.train_one_epoch(model, train_loader, optimizer, criterion)
                va_metrics = self.eval_one_epoch(model, val_loader, criterion)
                ep_sec = time.time() - ep_start

                row = {
                    "epoch": epoch + 1,
                    "timestamp_utc": utc_now_iso(),
                    "train_loss": tr_metrics["loss"],
                    "train_acc": tr_metrics["acc"],
                    "train_precision_macro": tr_metrics["precision_macro"],
                    "train_recall_macro": tr_metrics["recall_macro"],
                    "train_f1_macro": tr_metrics["f1_macro"],
                    "val_loss": va_metrics["loss"],
                    "val_acc": va_metrics["acc"],
                    "val_precision_macro": va_metrics["precision_macro"],
                    "val_recall_macro": va_metrics["recall_macro"],
                    "val_f1_macro": va_metrics["f1_macro"],
                }
                history_rows.append(row)
                append_jsonl(self.history_jsonl, row)

                current_val_metric = row[f"val_{best_metric_name}"]
                is_best = current_val_metric > best_metric_value
                if is_best:
                    best_metric_value = current_val_metric
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), self.best_model_path)
                    self.notifier.send_best_update(best_epoch, best_metric_name, best_metric_value)

                star = " ★ BEST" if is_best else ""
                self._log(
                    f"Época {epoch+1:>3}/{total_epochs} | "
                    f"train_loss={tr_metrics['loss']:.4f}  val_loss={va_metrics['loss']:.4f} | "
                    f"val_acc={va_metrics['acc']:.4f}  val_f1={va_metrics['f1_macro']:.4f} | "
                    f"{ep_sec:.1f}s{star}"
                )

                notify_every = self.cfg.get("discord", {}).get("notify_every_n_epochs", 1)
                if ((epoch + 1) % notify_every) == 0:
                    self.notifier.send_epoch(
                        epoch + 1,
                        total_epochs,
                        train_metrics=tr_metrics,
                        val_metrics=va_metrics
                    )

            hist_df = pd.DataFrame(history_rows)
            hist_df.to_csv(self.history_csv, index=False)

            self._log("═" * 50)
            self._log("FASE 4: EVALUACIÓN EN TEST")
            self._log("═" * 50)
            model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            test_metrics = self.eval_one_epoch(model, test_loader, criterion)
            self._log(
                f"Test: acc={test_metrics['acc']:.4f}  f1={test_metrics['f1_macro']:.4f}  "
                f"loss={test_metrics['loss']:.4f} (best_epoch={best_epoch})"
            )

            result = {
                "status": "completed",
                "experiment_name": self.experiment_name,
                "best_epoch": best_epoch,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric_value,
                "test_metrics": test_metrics,
                "output_dir": self.output_dir,
                "device": self.device,
                "num_studies": prep_summary["num_studies"],
                "num_vertebra_regions": prep_summary["num_vertebra_regions"],
                "num_subpatches": prep_summary["num_subpatches"],
            }

            save_json(self.result_json, result)
            self.notifier.send_result(result)

            total_min = (time.time() - self._t0) / 60
            self._log(f"✅ Experimento completado en {total_min:.1f} min")
            return result

        except Exception as e:
            self._log(f"❌ ERROR: {e}")
            err = traceback.format_exc()
            self.notifier.send_error(err)
            raise
