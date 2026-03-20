import logging
import json
from typing import Any

import torch
import torch.nn as nn
from injectable import injectable
from kafka import KafkaProducer

from dto.general_models import GeneralModels
from strategy.executer_model import ExecuterModel

logger = logging.getLogger("NeuralNetworkExecuter")

KAFKA_BROKER = "localhost:9092"


class _MLP(nn.Module):
    """Red neuronal densa configurable via hiperparámetros."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@injectable
class NeuralNetworkExecuter(ExecuterModel[GeneralModels]):

    def execute(self, model: GeneralModels) -> dict[str, Any]:
        hp = model.hyperparameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[NN] Dispositivo: {device} | Hiperparámetros: {hp}")

        net = _MLP(
            input_dim=hp.get("input_dim", 128),
            hidden_dim=hp.get("hidden_dim", 256),
            output_dim=hp.get("output_dim", 10),
            num_layers=hp.get("num_layers", 3),
        ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=hp.get("learning_rate", 1e-3))
        criterion = nn.CrossEntropyLoss()

        epochs = hp.get("epochs", 5)
        batch_size = hp.get("batch_size", 32)

        # Placeholder: datos sintéticos para demostración
        X = torch.randn(batch_size, hp.get("input_dim", 128), device=device)
        y = torch.randint(0, hp.get("output_dim", 10), (batch_size,), device=device)

        losses: list[float] = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = net(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            logger.info(f"[NN] Epoch {epoch + 1}/{epochs} — loss={loss.item():.4f}")

        result = {
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "epochs": epochs,
            "final_loss": losses[-1],
            "loss_history": losses,
            "device": device,
            "status": "trained",
        }
        logger.info(f"[NN] Entrenamiento completado: {result}")
        return result

    def publish_to_mlflow(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-mlflow", value=result)
            producer.flush()
            logger.info(f"[NN] Publicado en topic-mlflow: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[NN] Error publicando en topic-mlflow: {e}", exc_info=True)

    def publish_to_storage(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-storage", value=result)
            producer.flush()
            logger.info(f"[NN] Publicado en topic-storage: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[NN] Error publicando en topic-storage: {e}", exc_info=True)
