import json
import urllib.request


class DiscordWebhookNotifier:
    def __init__(self, webhook_url, experiment_name="experiment"):
        self.webhook_url = webhook_url
        self.experiment_name = experiment_name

    def _send(self, payload):
        data = json.dumps(payload).encode("utf-8")
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

    def send_result(self, result):
        title = f"✅ **{self.experiment_name}** — Entrenamiento finalizado"

        lines = [title, "```json"]
        lines.append(json.dumps(result, indent=2, ensure_ascii=False))
        lines.append("```")

        self._send({"content": "\n".join(lines)})

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
