from kafka import KafkaConsumer

KAFKA_BROKER = 'localhost:9092'
TOPIC = 'test-topic'

consumer = KafkaConsumer(TOPIC, bootstrap_servers=KAFKA_BROKER, auto_offset_reset='earliest', consumer_timeout_ms=2000)
print('Mensajes recibidos:')
for msg in consumer:
    print(msg.value.decode())
