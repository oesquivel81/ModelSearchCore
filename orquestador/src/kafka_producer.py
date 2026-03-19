from kafka import KafkaProducer

KAFKA_BROKER = 'localhost:9092'
TOPIC = 'test-topic'

producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)
producer.send(TOPIC, b'Hola, mundo!')
producer.flush()
print('Mensaje enviado: Hola, mundo!')
