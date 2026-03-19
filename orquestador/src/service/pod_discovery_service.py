from kubernetes import client, config
from typing import List, Any
import logging

logger = logging.getLogger("PodDiscoveryService")
logging.basicConfig(level=logging.INFO)

class PodDiscoveryService:
    def __init__(self, namespace: str = "default"):
        try:
            config.load_kube_config()
            logger.info("Kube config cargado localmente.")
        except Exception:
            config.load_incluster_config()
            logger.info("Kube config cargado desde el cluster.")
        self.v1 = client.CoreV1Api()
        self.namespace = namespace

    def get_pod_count(self) -> int:
        pods = self.v1.list_namespaced_pod(self.namespace)
        logger.info(f"Total pods en {self.namespace}: {len(pods.items)}")
        return len(pods.items)

    def get_pods(self) -> List[Any]:
        pods = self.v1.list_namespaced_pod(self.namespace)
        logger.info(f"Obtenidos {len(pods.items)} pods en {self.namespace}")
        return pods.items
