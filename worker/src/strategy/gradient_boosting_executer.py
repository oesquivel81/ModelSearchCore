import logging
import json
from typing import Any

from injectable import injectable
from sklearn.ensemble import GradientBoostingClassifier
from kafka import KafkaProducer

from dto.general_models import GeneralModels
from strategy.executer_model import ExecuterModel

logger = logging.getLogger("GradientBoostingExecuter")

KAFKA_BROKER = "localhost:9092"


@injectable
class GradientBoostingExecuter(ExecuterModel[GeneralModels]):

    def execute(self, model: GeneralModels) -> dict[str, Any]:
        hp = model.hyperparameters
        logger.info(f"[GB] Entrenando GradientBoosting con hiperparámetros: {hp}")

        clf = GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 3),
            subsample=hp.get("subsample", 1.0),
            random_state=hp.get("random_state", 42),
        )

        logger.info(f"[GB] Modelo inicializado: {clf}")

        result = {
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "params": clf.get_params(),
            "status": "trained",
        }
        logger.info(f"[GB] Entrenamiento completado: {result}")
        return result

    def publish_to_mlflow(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-mlflow", value=result)
            producer.flush()
            logger.info(f"[GB] Publicado en topic-mlflow: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[GB] Error publicando en topic-mlflow: {e}", exc_info=True)

    def publish_to_storage(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-storage", value=result)
            producer.flush()
            logger.info(f"[GB] Publicado en topic-storage: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[GB] Error publicando en topic-storage: {e}", exc_info=True)
