import json
from colab.src.MAIA_CANONICAL_COMBINATIONS_TDA.canonical_subpatch_visualizer import CanonicalSubpatchVisualizer

# Lee parámetros desde un archivo JSON externo
def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Ejemplo de archivo config.json:
# {
#     "img_path": "ruta/a/tu/imagen.png",
#     "patch_size": [128, 128],
#     "subpatch_size": [32, 32],
#     "stride": [32, 32],
#     "cols": 4
# }

if __name__ == "__main__":
    config = load_config("config.json")
    visualizer = CanonicalSubpatchVisualizer(
        config["img_path"],
        patch_size=tuple(config.get("patch_size", (128, 128))),
        subpatch_size=tuple(config.get("subpatch_size", (32, 32))),
        stride=tuple(config.get("stride", (32, 32)))
    )
    visualizer.show(cols=config.get("cols", 4))
