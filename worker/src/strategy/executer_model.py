from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from dto.general_models import GeneralModels

T = TypeVar("T", bound=GeneralModels)


class ExecuterModel(ABC, Generic[T]):

    @abstractmethod
    def execute(self, model: T) -> Any:
        """Ejecuta la estrategia del modelo con los hiperparámetros recibidos."""
        pass

    @abstractmethod
    def publish_to_mlflow(self, result: Any) -> None:
        """Publica el resultado del entrenamiento al tópico de MLflow."""
        pass

    @abstractmethod
    def publish_to_storage(self, result: Any) -> None:
        """Publica el resultado al tópico de almacenamiento (MongoDB o PostgreSQL)."""
        pass
