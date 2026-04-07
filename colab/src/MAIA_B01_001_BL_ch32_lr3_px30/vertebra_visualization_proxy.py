import cv2
import matplotlib.pyplot as plt
from colab.src.image_utils.vertebra_component_extractor import VertebraComponentExtractor

class VertebraVisualizationProxy:
        def show_individual_components(self, figsize=(6, 6)):
            """
            Visualiza cada componente (vértebra) individualmente con su bounding box y centroide.
            """
            for i, comp in enumerate(self.extractor.components):
                bbox = comp.get("bbox_tight", comp["bbox"])
                x1, y1, x2, y2 = bbox
                patch_img = self.img[y1:y2, x1:x2]
                cx = int(round(comp["centroid_x"])) - x1
                cy = int(round(comp["centroid_y"])) - y1
                plt.figure(figsize=figsize)
                plt.imshow(patch_img, cmap="gray")
                plt.scatter([cx], [cy], color="red", s=40, label="Centroide")
                plt.title(f"Componente {i} | Área: {comp['area']}")
                plt.legend()
                plt.axis("off")
                plt.show()
    def __init__(self, img_path, mask_path, min_area=150, pad_x=30, pad_y=15):
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None or self.mask is None:
            raise FileNotFoundError("No se pudo cargar la imagen o la máscara.")
        self.extractor = VertebraComponentExtractor(
            image=self.img,
            local_mask=self.mask,
            min_area=min_area,
            pad_x=pad_x,
            pad_y=pad_y
        ).run()
        self.extractor.build_adjusted_bboxes(pad_x_tight=8, pad_y_tight=6, top_extra_tight=10)

    def show_overlay_dual(self, figsize=(8, 12)):
        self.extractor.build_overlay_dual()
        self.extractor.show_overlay_dual(figsize=figsize)

    def show_centroid_curve(self, figsize=(8, 12)):
        centroides = [(comp["centroid_x"], comp["centroid_y"]) for comp in self.extractor.components]
        plt.figure(figsize=figsize)
        plt.imshow(self.img, cmap="gray")
        xs, ys = zip(*centroides)
        plt.plot(xs, ys, marker="o", color="lime", linewidth=2, markersize=8, label="Curva centroides")
        plt.scatter(xs, ys, color="red", s=40, label="Centroides")
        plt.legend()
        plt.title("Curva y centroides de vértebras")
        plt.axis("off")
        plt.show()

    def show_component_quality(self, **kwargs):
        self.extractor.print_component_quality(**kwargs)

    def show_quality_overlay(self, **kwargs):
        self.extractor.build_quality_overlay(**kwargs)
        self.extractor.show_quality_overlay()
