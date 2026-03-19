import threading
import logging
from kafka import KafkaConsumer
from dto.general_models import GeneralModels
from controllers.health import app
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WorkerMain")

KAFKA_BROKER = "localhost:9092"
TOPIC = "topic-jobs"


def process_message(raw: str) -> None:
    model = GeneralModels.model_validate_json(raw)
    logger.info(f"[PROCESSOR] Ejecutando modelo: {model.model_id} - tipo: {model.model_type}")
    logger.info(f"[PROCESSOR] Hiperparámetros: {model.hyperparameters}")
    # TODO: Aquí se invoca el StrategyDispatcher con el model


def kafka_listener():
    logger.info("=" * 50)
    logger.info(f"[LISTENER] Iniciando listener Kafka")
    logger.info(f"[LISTENER] Broker:  {KAFKA_BROKER}")
    logger.info(f"[LISTENER] Tópico:  {TOPIC}")
    logger.info(f"[LISTENER] Grupo:   worker-group")
    logger.info("=" * 50)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        group_id="worker-group"
    )

    logger.info("[LISTENER] Conexión establecida. Esperando mensajes...")

    for msg in consumer:
        try:
            raw = msg.value.decode()
            logger.info(f"[LISTENER] ─── Nuevo mensaje recibido ───")
            logger.info(f"[LISTENER] Partición : {msg.partition}")
            logger.info(f"[LISTENER] Offset    : {msg.offset}")
            logger.info(f"[LISTENER] Timestamp : {msg.timestamp}")
            logger.info(f"[LISTENER] Payload   : {raw}")
            process_message(raw)
        except Exception as e:
            logger.error(f"[LISTENER] Error al procesar mensaje: {e}", exc_info=True)


if __name__ == "__main__":
    listener_thread = threading.Thread(target=kafka_listener, daemon=True)
    listener_thread.start()

    logger.info("Iniciando servidor HTTP en puerto 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
