import logging
from typing import Any
from injectable import injectable

from dto.general_models import GeneralModels
from dto.model_type import ModelType
from strategy.executer_model import ExecuterModel

logger = logging.getLogger("StrategyDispatcher")


@injectable
class StrategyDispatcher:
    """
    Resuelve y ejecuta la estrategia correcta según el ModelType recibido.
    Sigue el patrón Strategy: cada tipo de modelo tiene su propio ExecuterModel.
    """

    def __init__(self) -> None:
        # Importación lazy para evitar dependencias circulares
        from strategy.random_forest_executer import RandomForestExecuter
        from strategy.gradient_boosting_executer import GradientBoostingExecuter
        from strategy.neural_network_executer import NeuralNetworkExecuter
        from strategy.svm_executer import SVMExecuter
        from strategy.linear_regression_executer import LinearRegressionExecuter

        self._registry: dict[ModelType, ExecuterModel] = {
            ModelType.RANDOM_FOREST: RandomForestExecuter(),
            ModelType.GRADIENT_BOOSTING: GradientBoostingExecuter(),
            ModelType.NEURAL_NETWORK: NeuralNetworkExecuter(),
            ModelType.SVM: SVMExecuter(),
            ModelType.LINEAR_REGRESSION: LinearRegressionExecuter(),
        }

    def dispatch(self, model: GeneralModels) -> Any:
        executer = self._registry.get(model.model_type)
        if executer is None:
            raise ValueError(f"[StrategyDispatcher] Tipo de modelo no soportado: {model.model_type}")

        logger.info(f"[StrategyDispatcher] Despachando → {model.model_type.value} (id={model.model_id})")
        result = executer.execute(model)
        executer.publish_to_mlflow(result)
        executer.publish_to_storage(result)
        return result
