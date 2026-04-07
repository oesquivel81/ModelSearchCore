import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .vertebra_patch_extractor import VertebraPatchExtractor
from .vertebra_patch_dataset_v2 import VertebraPatchDatasetV2
from .models import BaselinePatchCNN, VarianceInputPatchCNN, VarianceBranchPatchCNN
from utils.variance_patch_builder import VariancePatchBuilderV2
from utils.discord_webhook_notifier import DiscordWebhookNotifier


class VertebraVarianceExperimentV2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.output_dir = os.path.join(cfg["drive_root"], cfg["experiment_name"])
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        self._set_seed(cfg.get("seed", 42))

        self.extractor = VertebraPatchExtractor(
            base_dir=cfg["extractor"]["base_dir"],
            index_csv=cfg["extractor"]["index_csv"],
            image_col=cfg["extractor"].get("image_col", "radiograph_path"),
            mask_col=cfg["extractor"].get("mask_col", "label_binary_path"),
            split_col=cfg["extractor"].get("split_col", "split"),
            min_area=cfg["extractor"].get("min_area", 50),
            pad_x=cfg["extractor"].get("pad_x", 30),
            pad_y=cfg["extractor"].get("pad_y", 15),
            include_labels=cfg["extractor"].get("include_labels", ["good", "doubtful"]),
            save_root=os.path.join(self.output_dir, "extracted_patches")
        )

        self.builder = VariancePatchBuilderV2(
            patch_size=tuple(cfg["data"]["patch_size"]),
            variance_ksize=cfg.get("variance_ksize", 5),
            save_root=os.path.join(self.output_dir, "processed")
        )

        webhook_url = cfg.get("discord_webhook_url")
        if webhook_url:
            self.notifier = DiscordWebhookNotifier(
                webhook_url=webhook_url,
                experiment_name=cfg["experiment_name"]
            )
        else:
            self.notifier = None

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _split_df(self, df):
        if "split" in df.columns:
            train_df = df[df["split"] == "train"].copy()
            val_df = df[df["split"] == "val"].copy()
            test_df = df[df["split"] == "test"].copy()

            if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
                return train_df, val_df, test_df

        df = df.sample(frac=1.0, random_state=self.cfg["seed"]).reset_index(drop=True)
        n = len(df)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        return train_df, val_df, test_df

    def _build_model(self):
        model_type = self.cfg["model"]["type"]
        base_channels = self.cfg["model"].get("base_channels", 32)

        if model_type == "baseline":
            return BaselinePatchCNN(base_channels=base_channels, num_classes=13)
        elif model_type == "variance_input":
            return VarianceInputPatchCNN(base_channels=base_channels, num_classes=13)
        elif model_type == "variance_branch":
            return VarianceBranchPatchCNN(base_channels=base_channels, num_classes=13)
        else:
            raise ValueError(f"model_type no soportado: {model_type}")

    def _make_loader(self, df, shuffle=False):
        ds = VertebraPatchDatasetV2(
            metadata_df=df,
            builder=self.builder,
            model_type=self.cfg["model"]["type"]
        )
        return DataLoader(
            ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["training"].get("num_workers", 2)
        )

    def _run_epoch(self, model, loader, optimizer=None):
        train = optimizer is not None
        criterion = torch.nn.CrossEntropyLoss()

        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total = 0
        correct = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                if self.cfg["model"]["type"] == "variance_branch":
                    x_img = batch["image"].to(self.device)
                    x_var = batch["variance"].to(self.device)
                    y = batch["label"].to(self.device)
                    logits = model(x_img, x_var)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = model(x)

                loss = criterion(logits, y)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * y.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return {
            "loss": total_loss / max(1, total),
            "acc": correct / max(1, total)
        }

    # =========================================================
    # Gráficas y predicciones
    # =========================================================
    def _save_training_plots(self, history):
        """Guarda curvas de loss y accuracy como PNG."""
        epochs = [r["epoch"] for r in history]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, [r["train_loss"] for r in history], label="train_loss")
        axes[0].plot(epochs, [r["val_loss"] for r in history], label="val_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, [r["train_acc"] for r in history], label="train_acc")
        axes[1].plot(epochs, [r["val_acc"] for r in history], label="val_acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle(self.cfg["experiment_name"], fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _predict_all(self, model, loader):
        """Devuelve (all_preds, all_labels) del loader."""
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                if self.cfg["model"]["type"] == "variance_branch":
                    x_img = batch["image"].to(self.device)
                    x_var = batch["variance"].to(self.device)
                    y = batch["label"].to(self.device)
                    logits = model(x_img, x_var)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return np.array(all_preds), np.array(all_labels)

    def _save_confusion_matrix(self, preds, labels, num_classes=13):
        """Guarda matriz de confusión como imagen PNG."""
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for t, p in zip(labels, preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{self.cfg['experiment_name']} — Matriz de Confusión")

        ticks = list(range(num_classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_sample_predictions(self, model, loader, max_samples=16):
        """Guarda grilla con patches y sus etiquetas pred vs real."""
        model.eval()
        images_list = []
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in loader:
                if self.cfg["model"]["type"] == "variance_branch":
                    x_img = batch["image"].to(self.device)
                    x_var = batch["variance"].to(self.device)
                    y = batch["label"].to(self.device)
                    logits = model(x_img, x_var)
                    imgs = x_img
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = model(x)
                    imgs = x

                pred = torch.argmax(logits, dim=1)
                for j in range(imgs.size(0)):
                    images_list.append(imgs[j, 0].cpu().numpy())
                    preds_list.append(pred[j].item())
                    labels_list.append(y[j].item())
                    if len(images_list) >= max_samples:
                        break
                if len(images_list) >= max_samples:
                    break

        n = len(images_list)
        if n == 0:
            return None

        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        axes = np.array(axes).reshape(-1)

        for i in range(n):
            axes[i].imshow(images_list[i], cmap="gray")
            color = "green" if preds_list[i] == labels_list[i] else "red"
            axes[i].set_title(
                f"pred={preds_list[i]}  real={labels_list[i]}",
                color=color, fontsize=10
            )
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"{self.cfg['experiment_name']} — Predicciones", fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "sample_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def fit(self):
        metadata_df = self.extractor.extract_and_save()

        debug = self.cfg.get("extractor_debug", {})

        if debug.get("show_patches", False):
            self.extractor.show_patches(max_patches=debug.get("max_patches", 16))

        if debug.get("save_grid", False):
            grid_path = os.path.join(self.output_dir, "patch_grid.png")
            self.extractor.save_patch_grid(
                max_patches=debug.get("max_patches", 16),
                grid_path=grid_path,
            )

        if debug.get("save_metadata", False):
            sample_id = debug.get("sample_id", self.cfg["experiment_name"])
            self.extractor.save_patches_with_metadata(sample_id=sample_id)

        self.builder.save_outputs(metadata_df, max_samples=100)

        train_df, val_df, test_df = self._split_df(metadata_df)

        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader = self._make_loader(val_df, shuffle=False)
        test_loader = self._make_loader(test_df, shuffle=False)

        model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["training"]["lr"])

        best_metric = -1.0
        best_epoch = -1
        best_path = os.path.join(self.output_dir, "best_model.pt")
        history = []
        total_epochs = self.cfg["training"]["epochs"]

        for epoch in range(total_epochs):
            tr = self._run_epoch(model, train_loader, optimizer=optimizer)
            va = self._run_epoch(model, val_loader, optimizer=None)

            row = {
                "epoch": epoch + 1,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"]
            }
            history.append(row)
            print(row)

            if self.notifier:
                self.notifier.send_epoch(
                    epoch=epoch + 1,
                    total_epochs=total_epochs,
                    train_metrics=tr,
                    val_metrics=va
                )

            if va["acc"] > best_metric:
                best_metric = va["acc"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)

                if self.notifier:
                    self.notifier.send_best_update(best_epoch, "val_acc", best_metric)

        model.load_state_dict(torch.load(best_path, map_location=self.device))
        te = self._run_epoch(model, test_loader, optimizer=None)

        # --- Reportes configurables ---
        report = self.cfg.get("report", {})
        report_files = []

        if report.get("training_curves", True):
            curves_path = self._save_training_plots(history)
            report_files.append(curves_path)

        if report.get("confusion_matrix", True):
            test_preds, test_labels = self._predict_all(model, test_loader)
            cm_path = self._save_confusion_matrix(test_preds, test_labels)
            report_files.append(cm_path)

        if report.get("sample_predictions", True):
            sp_path = self._save_sample_predictions(
                model, test_loader,
                max_samples=report.get("max_prediction_samples", 16)
            )
            if sp_path:
                report_files.append(sp_path)

        if report.get("patch_grid", True) and len(self.extractor.components) > 0:
            grid_path = os.path.join(self.output_dir, "patch_grid.png")
            self.extractor.save_patch_grid(
                max_patches=report.get("max_patches", 20),
                grid_path=grid_path
            )
            report_files.append(grid_path)

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(os.path.join(self.output_dir, "history.csv"), index=False)

        result = {
            "status": "completed",
            "num_vertebras": len(metadata_df),
            "best_epoch": best_epoch,
            "best_metric_name": "val_acc",
            "best_metric_value": best_metric,
            "test_metrics": {
                "acc": te["acc"],
                "test_loss": te["loss"]
            },
            "output_dir": self.output_dir
        }

        with open(os.path.join(self.output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if self.notifier:
            self.notifier.send_result(result, file_paths=report_files)

        return result
