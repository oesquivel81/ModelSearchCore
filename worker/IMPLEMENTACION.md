# Cómo implementar un nuevo modelo en el Worker

Esta guía explica cómo agregar soporte para un nuevo tipo de modelo en el worker siguiendo el patrón Strategy.

---

## 1. Agregar el tipo de modelo al enum

Edita `src/dto/model_type.py` y agrega el nuevo tipo:

```python
class ModelType(str, Enum):
    ...
    XGBOOST = "xgboost"  # Nuevo tipo
```

---

## 2. Crear la clase de estrategia concreta

Crea un nuevo archivo en `src/strategy/`, por ejemplo `xgboost_executer.py`:

```python
from typing import Any
from strategy.executer_model import ExecuterModel
from dto.general_models import GeneralModels
from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092')

class XGBoostExecuter(ExecuterModel[GeneralModels]):

    def execute(self, model: GeneralModels) -> Any:
        params = model.hyperparameters
        # Aquí va la lógica real de entrenamiento
        print(f"Entrenando XGBoost con params: {params}")
        result = {"model_id": model.model_id, "accuracy": 0.97, "params": params}

        self.publish_to_mlflow(result)
        self.publish_to_storage(result)
        return result

    def publish_to_mlflow(self, result: Any) -> None:
        producer.send("topic-mlflow", json.dumps(result).encode())
        print(f"Publicado a MLflow: {result}")

    def publish_to_storage(self, result: Any) -> None:
        producer.send("topic-storage", json.dumps(result).encode())
        print(f"Publicado a Storage: {result}")
```

---

## 3. Registrar la estrategia en el dispatcher

Crea o edita un dispatcher en `src/service/` que mapee el `ModelType` a su estrategia correspondiente:

```python
from dto.model_type import ModelType
from strategy.xgboost_executer import XGBoostExecuter
from strategy.random_forest_executer import RandomForestExecuter

STRATEGY_MAP = {
    ModelType.XGBOOST: XGBoostExecuter(),
    ModelType.RANDOM_FOREST: RandomForestExecuter(),
}

def get_strategy(model_type: ModelType):
    strategy = STRATEGY_MAP.get(model_type)
    if not strategy:
        raise ValueError(f"No hay estrategia registrada para {model_type}")
    return strategy
```

---

## 4. Consumir desde Kafka y ejecutar

El consumer en `src/kafka_consumer.py` recibe el mensaje, lo encola y el worker lo procesa:

```python
from utils.hyperparameter_queue import HyperparameterQueue
from service.strategy_dispatcher import get_strategy

queue = HyperparameterQueue()

# Consumir desde Kafka
for msg in consumer:
    queue.enqueue_from_json_string(msg.value.decode())

# Procesar la cola (bloqueante)
while True:
    model = queue.dequeue()
    strategy = get_strategy(model.model_type)
    strategy.execute(model)
```

---

## Estructura de un mensaje JSON de entrada

```json
{
  "model_id": "abc-123",
  "model_type": "xgboost",
  "hyperparameters": {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6
  }
}
```

---

## Flujo completo

```
Orquestador → [Kafka: topic-jobs] → Worker Consumer
                                        ↓
                                 HyperparameterQueue
                                        ↓
                                 StrategyDispatcher
                                        ↓
                              ExecuterModel.execute()
                               ↙               ↘
                    [topic-mlflow]        [topic-storage]
                         ↓                      ↓
                       MLflow              MongoDB/PostgreSQL
```
