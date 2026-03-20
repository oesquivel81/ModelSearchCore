import sys
import os
import logging

# Permite importar los módulos de src/ sin instalar el paquete
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.hyperparameter_queue import HyperparameterQueue
from dto.job_config_dto import JobConfigDTO
from dto.model_type import ModelType
from dto.job_status import JobStatus

# ─────────────────────────────────────────────────────────────────────────────
#  Logger de tests
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestHyperparameterQueue")

# ─────────────────────────────────────────────────────────────────────────────
#  Payloads de prueba (los 6 mensajes que se enviarán a Kafka)
# ─────────────────────────────────────────────────────────────────────────────

JOBS = [
    {
        "model_type": "svm",
        "hyperparameters": {"C": 2.0, "kernel": "rbf", "gamma": "scale", "probability": True},
        "status": "QUEUED",
    },
    {
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5, "random_state": 42},
        "status": "QUEUED",
    },
    {
        "model_type": "gradient_boosting",
        "hyperparameters": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "random_state": 7},
        "status": "QUEUED",
    },
    {
        "model_type": "neural_network",
        "hyperparameters": {"input_dim": 128, "hidden_dim": 512, "output_dim": 10, "num_layers": 4, "learning_rate": 0.001, "epochs": 20, "batch_size": 64},
        "status": "QUEUED",
    },
    {
        "model_type": "svm",
        "hyperparameters": {"C": 2.0, "kernel": "rbf", "gamma": "scale", "probability": True},
        "status": "QUEUED",
    },
    {
        "model_type": "linear_regression",
        "hyperparameters": {"fit_intercept": True, "positive": False},
        "status": "QUEUED",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def queue_with_jobs() -> HyperparameterQueue:
    """Cola pre-cargada con los 6 jobs de prueba."""
    logger.info("━" * 60)
    logger.info("FIXTURE | Creando HyperparameterQueue con %d jobs", len(JOBS))
    q = HyperparameterQueue()
    for i, job in enumerate(JOBS, start=1):
        q.enqueue(job)
        logger.info("  [%d/%d] ✔ Encolado → model_type=%-20s  hp_count=%d",
                    i, len(JOBS), job["model_type"], len(job["hyperparameters"]))
    logger.info("FIXTURE | Cola lista — tamaño actual: %d", q.size())
    logger.info("━" * 60)
    return q


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnqueue:

    def test_queue_size_after_enqueue_all(self, queue_with_jobs):
        """La cola debe tener exactamente 6 elementos después de encolar todos."""
        size = queue_with_jobs.size()
        logger.info("TEST | queue_size_after_enqueue_all → tamaño=%d (esperado=6)", size)
        assert size == 6

    def test_queue_is_not_empty_after_enqueue(self, queue_with_jobs):
        empty = queue_with_jobs.is_empty()
        logger.info("TEST | queue_is_not_empty → is_empty=%s (esperado=False)", empty)
        assert not empty

    def test_enqueue_creates_job_config_dto_instances(self):
        """Cada item encolado debe ser una instancia válida de JobConfigDTO."""
        logger.info("TEST | enqueue_creates_job_config_dto_instances — encolando %d jobs", len(JOBS))
        q = HyperparameterQueue()
        for job in JOBS:
            q.enqueue(job)
        for i in range(len(JOBS)):
            item = q.dequeue(timeout=1)
            logger.info("  [%d] Desencolado → type=%-20s  class=%s",
                        i + 1, item.model_type.value, type(item).__name__)
            assert isinstance(item, JobConfigDTO)

    def test_enqueue_from_json_string(self):
        """enqueue_from_json_string debe parsear el JSON y encolar correctamente."""
        import json
        payload = json.dumps(JOBS[0])
        logger.info("TEST | enqueue_from_json_string — payload=%s", payload)
        q = HyperparameterQueue()
        q.enqueue_from_json_string(payload)
        logger.info("TEST | Cola tras enqueue_from_json_string → tamaño=%d", q.size())
        assert q.size() == 1
        item = q.dequeue(timeout=1)
        logger.info("TEST | Desencolado → model_type=%s", item.model_type.value)
        assert item.model_type == ModelType.SVM


class TestDequeue:

    def test_fifo_order(self, queue_with_jobs):
        """Los items deben desencolarse en orden FIFO."""
        logger.info("TEST | fifo_order — verificando orden de %d items", len(JOBS))
        expected_types = [ModelType(j["model_type"]) for j in JOBS]
        for i, expected in enumerate(expected_types, start=1):
            item = queue_with_jobs.dequeue(timeout=1)
            match = "✔" if item.model_type == expected else "✘"
            logger.info("  [%d] %s  dequeued=%-20s  expected=%-20s",
                        i, match, item.model_type.value, expected.value)
            assert item.model_type == expected

    def test_queue_empty_after_dequeue_all(self, queue_with_jobs):
        """La cola debe estar vacía después de desencolar todos los elementos."""
        total = queue_with_jobs.size()
        logger.info("TEST | queue_empty_after_dequeue_all — vaciando %d items", total)
        for i in range(total):
            item = queue_with_jobs.dequeue(timeout=1)
            logger.info("  [%d/%d] Consumido → %-20s  (restantes=%d)",
                        i + 1, total, item.model_type.value, queue_with_jobs.size())
        logger.info("TEST | Cola vacía: is_empty=%s", queue_with_jobs.is_empty())
        assert queue_with_jobs.is_empty()


class TestJobFields:

    def test_all_jobs_have_status_queued(self, queue_with_jobs):
        """Todos los jobs deben tener status QUEUED."""
        logger.info("TEST | all_jobs_have_status_queued — revisando %d items", queue_with_jobs.size())
        for i in range(queue_with_jobs.size()):
            item = queue_with_jobs.dequeue(timeout=1)
            logger.info("  [%d] model=%-20s  status=%s", i + 1, item.model_type.value, item.status.value)
            assert item.status == JobStatus.QUEUED

    def test_svm_hyperparameters(self):
        logger.info("TEST | svm_hyperparameters")
        q = HyperparameterQueue()
        q.enqueue(JOBS[0])
        item = q.dequeue(timeout=1)
        logger.info("  model_type=%-20s  C=%s  kernel=%s  gamma=%s  probability=%s",
                    item.model_type.value,
                    item.hyperparameters["C"],
                    item.hyperparameters["kernel"],
                    item.hyperparameters["gamma"],
                    item.hyperparameters["probability"])
        assert item.model_type == ModelType.SVM
        assert item.hyperparameters["C"] == 2.0
        assert item.hyperparameters["kernel"] == "rbf"
        assert item.hyperparameters["gamma"] == "scale"
        assert item.hyperparameters["probability"] is True

    def test_random_forest_hyperparameters(self):
        logger.info("TEST | random_forest_hyperparameters")
        q = HyperparameterQueue()
        q.enqueue(JOBS[1])
        item = q.dequeue(timeout=1)
        logger.info("  model_type=%-20s  n_estimators=%s  max_depth=%s  min_samples_split=%s  random_state=%s",
                    item.model_type.value,
                    item.hyperparameters["n_estimators"],
                    item.hyperparameters["max_depth"],
                    item.hyperparameters["min_samples_split"],
                    item.hyperparameters["random_state"])
        assert item.model_type == ModelType.RANDOM_FOREST
        assert item.hyperparameters["n_estimators"] == 200
        assert item.hyperparameters["max_depth"] == 10
        assert item.hyperparameters["min_samples_split"] == 5
        assert item.hyperparameters["random_state"] == 42

    def test_gradient_boosting_hyperparameters(self):
        logger.info("TEST | gradient_boosting_hyperparameters")
        q = HyperparameterQueue()
        q.enqueue(JOBS[2])
        item = q.dequeue(timeout=1)
        logger.info("  model_type=%-20s  n_estimators=%s  learning_rate=%s  max_depth=%s  subsample=%s",
                    item.model_type.value,
                    item.hyperparameters["n_estimators"],
                    item.hyperparameters["learning_rate"],
                    item.hyperparameters["max_depth"],
                    item.hyperparameters["subsample"])
        assert item.model_type == ModelType.GRADIENT_BOOSTING
        assert item.hyperparameters["learning_rate"] == pytest.approx(0.05)
        assert item.hyperparameters["subsample"] == pytest.approx(0.8)

    def test_neural_network_hyperparameters(self):
        logger.info("TEST | neural_network_hyperparameters")
        q = HyperparameterQueue()
        q.enqueue(JOBS[3])
        item = q.dequeue(timeout=1)
        logger.info("  model_type=%-20s  input_dim=%s  hidden_dim=%s  epochs=%s  batch_size=%s  lr=%s",
                    item.model_type.value,
                    item.hyperparameters["input_dim"],
                    item.hyperparameters["hidden_dim"],
                    item.hyperparameters["epochs"],
                    item.hyperparameters["batch_size"],
                    item.hyperparameters["learning_rate"])
        assert item.model_type == ModelType.NEURAL_NETWORK
        assert item.hyperparameters["input_dim"] == 128
        assert item.hyperparameters["hidden_dim"] == 512
        assert item.hyperparameters["epochs"] == 20
        assert item.hyperparameters["batch_size"] == 64

    def test_linear_regression_hyperparameters(self):
        logger.info("TEST | linear_regression_hyperparameters")
        q = HyperparameterQueue()
        q.enqueue(JOBS[5])
        item = q.dequeue(timeout=1)
        logger.info("  model_type=%-20s  fit_intercept=%s  positive=%s",
                    item.model_type.value,
                    item.hyperparameters["fit_intercept"],
                    item.hyperparameters["positive"])
        assert item.model_type == ModelType.LINEAR_REGRESSION
        assert item.hyperparameters["fit_intercept"] is True
        assert item.hyperparameters["positive"] is False

    def test_duplicate_model_types_allowed(self, queue_with_jobs):
        """La cola permite múltiples jobs del mismo tipo (2 x SVM en el fixture)."""
        logger.info("TEST | duplicate_model_types_allowed — contando SVMs en la cola")
        svm_count = 0
        for i in range(queue_with_jobs.size()):
            item = queue_with_jobs.dequeue(timeout=1)
            if item.model_type == ModelType.SVM:
                svm_count += 1
                logger.info("  SVM encontrado en posición %d (total hasta ahora: %d)", i + 1, svm_count)
        logger.info("  Total SVMs: %d (esperado=2)", svm_count)
        assert svm_count == 2


class TestEdgeCases:

    def test_empty_queue_is_empty(self):
        logger.info("TEST | empty_queue_is_empty")
        q = HyperparameterQueue()
        logger.info("  is_empty=%s  size=%d", q.is_empty(), q.size())
        assert q.is_empty()
        assert q.size() == 0

    def test_invalid_model_type_raises(self):
        logger.info("TEST | invalid_model_type_raises — enviando model_type='invalid_type'")
        q = HyperparameterQueue()
        with pytest.raises(Exception) as exc_info:
            q.enqueue({"model_type": "invalid_type", "hyperparameters": {}, "status": "QUEUED"})
        logger.info("  Excepción capturada correctamente: %s", exc_info.type.__name__)

    def test_missing_model_type_raises(self):
        logger.info("TEST | missing_model_type_raises — payload sin 'model_type'")
        q = HyperparameterQueue()
        with pytest.raises(Exception) as exc_info:
            q.enqueue({"hyperparameters": {}, "status": "QUEUED"})
        logger.info("  Excepción capturada correctamente: %s", exc_info.type.__name__)
