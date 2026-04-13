import json
from extractor.ablation_pipeline_proxy import AblationPipelineProxy

# Cargar configuración desde archivo JSON
with open('colab/src/extractor/ablation_pipeline_config_example.json', 'r') as f:
    config = json.load(f)

# Ejecutar el pipeline completo (centroides + ablation)
proxy = AblationPipelineProxy(config)
proxy.run()

# Para solo centroides y parches (sin ablation), puedes usar:
# from extractor.centroid_curve_proxy import CentroidCurveProxy
# with open('colab/src/extractor/centroid_curve_config_full_example.json', 'r') as f:
#     config_centroid = json.load(f)
# proxy_centroid = CentroidCurveProxy(config_centroid)
# proxy_centroid.run_all()
