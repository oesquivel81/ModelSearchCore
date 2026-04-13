import json
from extractor.centroid_curve_proxy import CentroidCurveProxy
from MAIA_B01_001_BL_FILTERS.patch_ablation_runner import PatchAblationRunner, AblationConfig
from extractor.vertebra_region_extractor import VertebraAutoCentroidExtractor
from extractor.patch_metrics import PatchMetrics
from extractor.patch_dto import PatchDTOBuilder
from itertools import product

class AblationPipelineProxy:
    def __init__(self, config):
        self.config = config
        self.patient_id = config["patient_id"]
        self.save_root = config["save_root"]
        self.all_filter_combinations = config.get("all_filter_combinations", True)
        self.selected_filters = config.get("selected_filters", [])
        self.centroid_proxy = CentroidCurveProxy(config)

    def generar_filtro_pipelines_coherentes(self, all=True, selected_filters=None):
        if not all and selected_filters is not None:
            return sorted(set(selected_filters))
        pipelines = set()
        base = ["none"]
        contrast = ["clahe"]
        smooth = ["gaussian", "median", "bilateral"]
        sharpen = ["unsharp_mask"]
        edge = ["sobel", "scharr", "prewitt", "laplacian", "log", "canny"]
        special = ["local_variance"]
        pipelines.update(base)
        pipelines.update(contrast)
        pipelines.update(smooth)
        pipelines.update(sharpen)
        pipelines.update(edge)
        pipelines.update(special)
        for c, s in product(contrast, smooth):
            pipelines.add(f"{c}+{s}")
        for s, e in product(smooth, edge):
            pipelines.add(f"{s}+{e}")
        for c, e in product(contrast, edge):
            pipelines.add(f"{c}+{e}")
        for c, s, e in product(contrast, smooth, edge):
            pipelines.add(f"{c}+{s}+{e}")
        for s, u in product(smooth, sharpen):
            pipelines.add(f"{s}+{u}")
        for c, u in product(contrast, sharpen):
            pipelines.add(f"{c}+{u}")
        for c, s, u in product(contrast, smooth, sharpen):
            pipelines.add(f"{c}+{s}+{u}")
        for c, s, u, e in product(contrast, smooth, sharpen, edge):
            pipelines.add(f"{c}+{s}+{u}+{e}")
        pipelines.add("clahe+local_variance")
        pipelines.add("gaussian+local_variance")
        pipelines.add("median+local_variance")
        pipelines.add("bilateral+local_variance")
        pipelines.add("clahe+gaussian+local_variance")
        pipelines.add("clahe+median+local_variance")
        pipelines.add("clahe+bilateral+local_variance")
        return sorted(pipelines)

    def run(self):
        # 1. Generar parches (centroides)
        self.centroid_proxy.run_all()
        # 2. Ejecutar ablation de filtros
        base_path = self.config["base_path"]
        image_path = self.config["img_rel_path"]
        mask_path = self.config["mask_rel_path"]
        extractor = VertebraAutoCentroidExtractor(
            base_dir=base_path,
            image_col="radiograph_path",
            mask_col="label_binary_path"
        )
        image = extractor._read_gray_rel(image_path)
        mask = extractor._read_gray_rel(mask_path)
        if self.all_filter_combinations:
            filtros = self.generar_filtro_pipelines_coherentes(all=True)
            use_variance_opts = [False, True]
            variance_modes = ["none", "concat_channel", "variance_only"]
            patch_sizes = [(128, 128), (256, 256)]
            strides = [32, 64]
            configs = []
            for i, (f, u, v, p, s) in enumerate(product(filtros, use_variance_opts, variance_modes, patch_sizes, strides)):
                configs.append(
                    AblationConfig(
                        config_id=f"cfg_{i:05d}",
                        filter_name=f,
                        use_variance=u,
                        variance_mode=v,
                        patch_size=p,
                        stride=s
                    )
                )
        else:
            configs = [
                AblationConfig(
                    config_id=f"cfg_{i:05d}",
                    filter_name=cf["filter_name"],
                    use_variance=cf["use_variance"],
                    variance_mode=cf["variance_mode"],
                    patch_size=tuple(cf["patch_size"]),
                    stride=cf["stride"]
                )
                for i, cf in enumerate(self.selected_filters)
            ]
        patch_builder = PatchDTOBuilder(save_root=self.save_root)
        metrics = PatchMetrics(kernel_size=3, hausdorff_use_edges=True)
        dataset = [{
            "image": image,
            "mask": mask,
            "patient_id": self.patient_id
        }]
        runner = PatchAblationRunner(extractor, patch_builder, metrics)
        df_results = runner.run_all(dataset, configs)
        df_results.to_csv(f"{self.save_root}/metrics_ablation_{self.patient_id}.csv", index=False)
        print(f"¡Ablation terminada y métricas guardadas para {self.patient_id}!")
