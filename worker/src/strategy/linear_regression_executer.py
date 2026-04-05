import logging
import json
from typing import Any

from injectable import injectable
from sklearn.linear_model import LinearRegression
from kafka import KafkaProducer

from dto.general_models import GeneralModels
from strategy.executer_model import ExecuterModel

logger = logging.getLogger("LinearRegressionExecuter")

KAFKA_BROKER = "localhost:9092"


@injectable
class LinearRegressionExecuter(ExecuterModel[GeneralModels]):

    def execute(self, model: GeneralModels) -> dict[str, Any]:
        hp = model.hyperparameters
        logger.info(f"[LR] Entrenando LinearRegression con hiperparámetros: {hp}")

        reg = LinearRegression(
            fit_intercept=hp.get("fit_intercept", True),
            positive=hp.get("positive", False),
        )

        logger.info(f"[LR] Modelo inicializado: {reg}")

        ##IMPLEMENTAR AQUÍ EL ENTRENAMIENTO REAL CON LOS DATOS REALES


        result = {
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "params": reg.get_params(),
            "status": "trained",
        }
        logger.info(f"[LR] Entrenamiento completado: {result}")
        return result

    def publish_to_mlflow(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-mlflow", value=result)
            producer.flush()
            logger.info(f"[LR] Publicado en topic-mlflow: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[LR] Error publicando en topic-mlflow: {e}", exc_info=True)

    def publish_to_storage(self, result: dict[str, Any]) -> None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send("topic-storage", value=result)
            producer.flush()
            logger.info(f"[LR] Publicado en topic-storage: model_id={result['model_id']}")
        except Exception as e:
            logger.error(f"[LR] Error publicando en topic-storage: {e}", exc_info=True)
