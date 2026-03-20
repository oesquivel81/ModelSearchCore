import logging
import json
from typing import Any

from injectable import injectable
from sklearn.svm import SVC
from kafka import KafkaProducer

from dto.general_models import GeneralModels
from strategy.executer_model import ExecuterModel

logger = logging.getLogger("SVMExecuter")

KAFKA_BROKER = "localhost:9092"


@injectable
class SVMExecuter(ExecuterModel[GeneralModels]):

    def execute(self, model: GeneralModels) -> dict[str, Any]:
        hp = model.hyperparameters
        logger.info(f"[SVM] Entrenando SVM con hiperparámetros: {hp}")

        clf = SVC(
            C=hp.get("C", 1.0),
            kernel=hp.get("kernel", "rbf"),
            gamma=hp.get("gamma", "scale"),
            probability=hp.get("probability", True),
        )

        logger.info(f"[SVM] Modelo inicializado: {clf}")

        result = {
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "params": clf.get_params(),
            "status": "trained",
        }
        logger.info(f"[SVM] Entrenamiento completado: {result}")
        return result

    def publish_to_mlflow(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-mlflow", value=result)
            producer.flush()
            logger.info(f"[SVM] Publicado en topic-mlflow: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[SVM] Error publicando en topic-mlflow: {e}", exc_info=True)

    def publish_to_storage(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-storage", value=result)
            producer.flush()
            logger.info(f"[SVM] Publicado en topic-storage: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[SVM] Error publicando en topic-storage: {e}", exc_info=True)
