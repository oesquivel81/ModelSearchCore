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
