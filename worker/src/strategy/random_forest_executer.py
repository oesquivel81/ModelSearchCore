import logging
import json
from typing import Any

from injectable import injectable
from sklearn.ensemble import RandomForestClassifier
from kafka import KafkaProducer

from dto.general_models import GeneralModels
from strategy.executer_model import ExecuterModel

logger = logging.getLogger("RandomForestExecuter")

KAFKA_BROKER = "localhost:9092"


@injectable
class RandomForestExecuter(ExecuterModel[GeneralModels]):

    def execute(self, model: GeneralModels) -> dict[str, Any]:
        hp = model.hyperparameters
        logger.info(f"[RF] Entrenando RandomForest con hiperparámetros: {hp}")

        clf = RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            random_state=hp.get("random_state", 42),
        )

        # Placeholder: en producción se cargarian los datos reales
        logger.info(f"[RF] Modelo inicializado: {clf}")

        result = {
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "params": clf.get_params(),
            "status": "trained",
        }
        logger.info(f"[RF] Entrenamiento completado: {result}")
        return result

    def publish_to_mlflow(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-mlflow", value=result)
            producer.flush()
            logger.info(f"[RF] Publicado en topic-mlflow: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[RF] Error publicando en topic-mlflow: {e}", exc_info=True)

    def publish_to_storage(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-storage", value=result)
            producer.flush()
            logger.info(f"[RF] Publicado en topic-storage: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[RF] Error publicando en topic-storage: {e}", exc_info=True)
