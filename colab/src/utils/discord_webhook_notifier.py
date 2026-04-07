import json
import os
import urllib.request

try:
    import requests as _requests
except ImportError:
    _requests = None


class DiscordWebhookNotifier:
    def __init__(self, webhook_url, experiment_name="experiment"):
        self.webhook_url = webhook_url
        self.experiment_name = experiment_name

    def _send(self, payload, file_paths=None):
        if not self.webhook_url:
            return

        content = payload.get("content", "")[:2000]

        if file_paths and _requests is not None:
            self._send_with_files(content, file_paths)
            return

        if _requests is not None:
            try:
                _requests.post(
                    self.webhook_url,
                    json={"content": content},
                    timeout=30,
                )
            except Exception as e:
                print(f"[DiscordWebhook] Error enviando mensaje: {e}")
            return

        data = json.dumps({"content": content}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[DiscordWebhook] Error enviando mensaje: {e}")

    def _send_with_files(self, content, file_paths):
        opened = []
        files = []
        try:
            for i, fp in enumerate(file_paths):
                if not os.path.isfile(fp):
                    continue
                fh = open(fp, "rb")
                opened.append(fh)
                files.append((f"file{i}", (os.path.basename(fp), fh, "image/png")))

            _requests.post(
                self.webhook_url,
                data={"payload_json": json.dumps({"content": content[:2000]})},
                files=files,
                timeout=60,
            )
        except Exception as e:
            print(f"[DiscordWebhook] Error enviando con archivos: {e}")
        finally:
            for fh in opened:
                fh.close()

    def send_text(self, text):
        self._send({"content": text})

    def send_epoch(self, epoch, total_epochs, train_metrics, val_metrics):
        title = f"**{self.experiment_name}** — Época {epoch}/{total_epochs}"

        lines = [title, "```"]
        for k, v in train_metrics.items():
            lines.append(f"  train_{k}: {v:.4f}")
        for k, v in val_metrics.items():
            lines.append(f"  val_{k}:   {v:.4f}")
        lines.append("```")

        self._send({"content": "\n".join(lines)})

    def send_best_update(self, epoch, metric_name, metric_value):
        text = (
            f"🏆 **{self.experiment_name}** — Nuevo mejor modelo\n"
            f"  época {epoch}  |  {metric_name} = {metric_value:.4f}"
        )
        self._send({"content": text})

    def send_result(self, result, file_paths=None):
        title = f"✅ **{self.experiment_name}** — Entrenamiento finalizado"

        lines = [title, "```json"]
        lines.append(json.dumps(result, indent=2, ensure_ascii=False))
        lines.append("```")

        self._send({"content": "\n".join(lines)}, file_paths=file_paths)

    def send_error(self, error_msg):
        text = f"❌ **{self.experiment_name}** — Error\n```\n{error_msg}\n```"
        self._send({"content": text})

    def send_grid_start(self, total, batch_summary):
        lines = [
            f"🔬 **GRID SEARCH** — {total} experimentos",
            "```",
        ]
        for row in batch_summary:
            lines.append(
                f"  [{row['id']:>3}] {row['model_type']:<18} "
                f"ps={row['patch_size']}  padx={row['pad_x']}  "
                f"pady={row['pad_y']}  lr={row['lr']}"
            )
        lines.append("```")
        self._send({"content": "\n".join(lines)})

    def send_grid_row(self, row_info):
        status = row_info.get("status", "completed")
        exp_id = row_info.get("id", "?")
        name = row_info.get("experiment_name", "")

        if status == "completed":
            n_vert = row_info.get("num_vertebras", "?")
            val_acc = row_info.get("best_metric_value", 0)
            test_acc = row_info.get("test_acc", 0)
            text = (
                f"✅ **[{exp_id}]** {name}\n"
                f"```\n"
                f"  vertebras: {n_vert}  |  val_acc: {val_acc:.4f}  |  "
                f"test_acc: {test_acc:.4f}\n"
                f"```"
            )
        else:
            err = row_info.get("error", "desconocido")
            text = f"❌ **[{exp_id}]** {name}\n```\n  {err}\n```"
        self._send({"content": text})

    def send_grid_summary(self, results_df, csv_path):
        ok = len(results_df[results_df["status"] == "completed"])
        fail = len(results_df) - ok
        total = len(results_df)

        lines = [
            f"📊 **GRID FINALIZADO** — {ok}/{total} OK  |  {fail} FAIL",
            "```",
            f"{'ID':>4}  {'Modelo':<18} {'Patch':>7} {'PadX':>4} "
            f"{'PadY':>4} {'LR':>8} {'Vert':>5} {'ValAcc':>7} {'TestAcc':>7}",
            "-" * 85,
        ]

        completed = results_df[results_df["status"] == "completed"].sort_values(
            "test_acc", ascending=False
        )
        for _, r in completed.iterrows():
            lines.append(
                f"{int(r.get('id', 0)):>4}  {r['model_type']:<18} "
                f"{r['patch_size']:>7} {r['pad_x']:>4} {r['pad_y']:>4} "
                f"{r['lr']:>8.1e} {int(r.get('num_vertebras', 0)):>5} "
                f"{r.get('best_metric_value', 0):>7.4f} "
                f"{r.get('test_acc', 0):>7.4f}"
            )

        lines.append("```")
        lines.append(f"📁 CSV: `{csv_path}`")

        msg = "\n".join(lines)
        if len(msg) > 1950:
            msg = msg[:1950] + "\n...truncado"
        self._send({"content": msg})

    # ---------------------------------------------------------
    # UNet region experiment notifications (con imágenes)
    # ---------------------------------------------------------
    def send_unet_epoch(self, epoch, total_epochs, train_result, val_metrics,
                        is_best=False, best_metric_name="dice", file_paths=None):
        star = " ⭐ BEST" if is_best else ""
        msg = (
            f"**{self.experiment_name}** | Epoch {epoch}/{total_epochs}{star}\n"
            f"```\n"
            f"train_loss : {train_result['train_loss']:.4f}\n"
            f"val_loss   : {val_metrics['val_loss']:.4f}\n"
            f"dice       : {val_metrics['dice']:.4f}\n"
            f"iou        : {val_metrics['iou']:.4f}\n"
            f"precision  : {val_metrics['precision']:.4f}\n"
            f"recall     : {val_metrics['recall']:.4f}\n"
            f"f1         : {val_metrics['f1']:.4f}\n"
            f"hausdorff  : {val_metrics['hausdorff']:.4f}\n"
            f"speed      : {train_result['samples_per_sec']:.1f} samples/s\n"
            f"```"
        )
        self._send({"content": msg}, file_paths=file_paths)

    def send_unet_complete(self, summary, best_metric_name="dice", file_paths=None):
        tm = summary.get("test_metrics", {})
        msg = (
            f"✅ **{self.experiment_name}** — Entrenamiento completado\n"
            f"```\n"
            f"best epoch   : {summary.get('best_epoch')}\n"
            f"best {best_metric_name:9s}: {summary.get('best_metric_value', 0):.4f}\n"
            f"test dice    : {tm.get('dice', 0):.4f}\n"
            f"test iou     : {tm.get('iou', 0):.4f}\n"
            f"test f1      : {tm.get('f1', 0):.4f}\n"
            f"test hausd.  : {tm.get('hausdorff', 0):.4f}\n"
            f"duration     : {summary.get('experiment_duration_sec', 0):.0f}s\n"
            f"device       : {summary.get('device')}\n"
            f"patches      : {summary.get('train_patches')}/{summary.get('val_patches')}/{summary.get('test_patches')}\n"
            f"```"
        )
        self._send({"content": msg}, file_paths=file_paths)
