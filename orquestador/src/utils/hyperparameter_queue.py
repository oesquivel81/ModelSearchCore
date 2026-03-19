import queue
import json
import logging
from typing import Optional
from dto.job_config_dto import JobConfigDTO

logger = logging.getLogger("HyperparameterQueue")
logging.basicConfig(level=logging.INFO)


class HyperparameterQueue:

    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[JobConfigDTO] = queue.Queue(maxsize=maxsize)

    def enqueue(self, raw_json: dict) -> None:
        job = JobConfigDTO(**raw_json)
        self._queue.put(job)
        logger.info(f"Encolado job de tipo {job.model_type} con {len(job.hyperparameters)} hiperparámetros")

    def enqueue_from_json_string(self, json_str: str) -> None:
        raw = json.loads(json_str)
        self.enqueue(raw)

    def dequeue(self, timeout: Optional[float] = None) -> JobConfigDTO:
        job = self._queue.get(block=True, timeout=timeout)
        logger.info(f"Desencolado job de tipo {job.model_type}")
        return job

    def size(self) -> int:
        return self._queue.qsize()

    def is_empty(self) -> bool:
        return self._queue.empty()
