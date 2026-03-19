import queue
import json
import logging
from typing import Optional
from dto.general_models import GeneralModels

logger = logging.getLogger("HyperparameterQueue")
logging.basicConfig(level=logging.INFO)


class HyperparameterQueue:

    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[GeneralModels] = queue.Queue(maxsize=maxsize)

    def enqueue(self, raw_json: dict) -> None:
        model = GeneralModels(**raw_json)
        self._queue.put(model)
        logger.info(f"Encolado modelo {model.model_id} de tipo {model.model_type}")

    def dequeue(self, timeout: Optional[float] = None) -> GeneralModels:
        model = self._queue.get(block=True, timeout=timeout)
        logger.info(f"Desencolado modelo {model.model_id} de tipo {model.model_type}")
        return model

    def enqueue_from_json_string(self, json_str: str) -> None:
        raw = json.loads(json_str)
        self.enqueue(raw)

    def size(self) -> int:
        return self._queue.qsize()

    def is_empty(self) -> bool:
        return self._queue.empty()
