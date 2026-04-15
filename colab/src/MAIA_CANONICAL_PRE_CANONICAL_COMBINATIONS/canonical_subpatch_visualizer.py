import cv2
import matplotlib.pyplot as plt
from colab.src.extractor.vertebra_subpatch_generator import VertebraSubpatchGenerator

class CanonicalSubpatchVisualizer:
    def __init__(self, img_path, patch_size=(128, 128), subpatch_size=(32, 32), stride=(32, 32)):
        self.img_path = img_path
        self.patch_size = patch_size
        self.subpatch_size = subpatch_size
        self.stride = stride
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
        self.subpatch_gen = VertebraSubpatchGenerator(
            patch_size=patch_size,
            subpatch_size=subpatch_size,
            stride=stride
        )
        self.img_patch = self.subpatch_gen._resize(self.img)
        self.windows = self.subpatch_gen._extract_windows(self.img_patch)

    def show(self, cols=4):
        n = len(self.windows)
        rows = int(n / cols) + (n % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten()
        for i, (r, c, x1, y1, x2, y2, sub) in enumerate(self.windows):
            axes[i].imshow(sub, cmap="gray")
            axes[i].set_title(f"Subpatch {i}")
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    def show_multiple_resolutions(self, patch_sizes, subpatch_sizes, stride=(32, 32), cols=4):
        """
        Visualiza subpatches para varias resoluciones de patch y subpatch.
        patch_sizes: lista de tuplas (ej: [(128,128), (256,256)])
        subpatch_sizes: lista de tuplas (ej: [(32,32), (64,64)])
        """
        for patch_size in patch_sizes:
            for subpatch_size in subpatch_sizes:
                subpatch_gen = VertebraSubpatchGenerator(
                    patch_size=patch_size,
                    subpatch_size=subpatch_size,
                    stride=stride
                )
                img_patch = subpatch_gen._resize(self.img)
                windows = subpatch_gen._extract_windows(img_patch)
                n = len(windows)
                rows = int(n / cols) + (n % cols > 0)
                fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
                axes = axes.flatten()
                for i, (r, c, x1, y1, x2, y2, sub) in enumerate(windows):
                    axes[i].imshow(sub, cmap="gray")
                    axes[i].set_title(f"{patch_size}/{subpatch_size} - {i}")
                    axes[i].axis("off")
                for i in range(n, len(axes)):
                    axes[i].axis("off")
        plt.tight_layout()
        plt.show()
