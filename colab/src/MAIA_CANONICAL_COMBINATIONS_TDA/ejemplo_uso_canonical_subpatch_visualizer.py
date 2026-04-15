from colab.src.MAIA_CANONICAL_COMBINATIONS_TDA.canonical_subpatch_visualizer import CanonicalSubpatchVisualizer

# Cambia esta ruta por la ruta real de tu imagen
img_path = "ruta/a/tu/imagen.png"

# Instancia el visualizador con los tamaños deseados
test_visualizer = CanonicalSubpatchVisualizer(
    img_path,
    patch_size=(128, 128),
    subpatch_size=(32, 32),
    stride=(32, 32)
)

test_visualizer.show(cols=4)

# Para visualizar múltiples resoluciones:
# test_visualizer.show_multiple_resolutions(
#     patch_sizes=[(128,128), (256,256)],
#     subpatch_sizes=[(32,32), (64,64)],
#     stride=(32,32),
#     cols=4
# )
