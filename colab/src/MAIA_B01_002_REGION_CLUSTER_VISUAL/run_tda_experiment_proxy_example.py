import json
from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_experiment_proxy import TDAExperimentProxy

# Cargar configuración desde archivo JSON
with open('colab/src/MAIA_B01_002_REGION_CLUSTER_VISUAL/tda_experiment_config_example.json', 'r') as f:
    config = json.load(f)

# Ejecutar el pipeline TDA/curva/nerve
proxy = TDAExperimentProxy(config)
proxy.run()
