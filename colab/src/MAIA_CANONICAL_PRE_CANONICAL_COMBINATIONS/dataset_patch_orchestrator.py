import os
import json
import cv2
import pandas as pd
from .apply_filter_chain import apply_filter_chain


class DatasetPatchOrchestrator:
    def __init__(self, config, truncate=None, mask_columns_override=None):
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            raise TypeError("config debe ser dict o path a JSON")

        self.dataset_csv = self.config["dataset_csv"]
        self.save_root = self.config["save_root"]
        self.filters = self.config.get("filters", ["none"])
        self.patch_size = tuple(self.config.get("patch_size", (128, 128)))
        self.stride = self.config.get("stride", 32)
        self.centroid_curve_dir = self.config.get("centroid_curve_dir", self.save_root)
        self.dir_root = self.config.get("dir_root", None)

        self.truncate = truncate
        self.mask_columns_override = mask_columns_override

        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.centroid_curve_dir, exist_ok=True)

    def _resolve_path(self, path_value):
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        if self.dir_root and not os.path.isabs(path_value):
            return os.path.join(self.dir_root, path_value)
        return path_value

    def _get_mask_columns(self):
        if self.mask_columns_override is not None:
            return self.mask_columns_override
        return [
            "mask_path",
            "label_binary_path",
            "multiclass_id_png",
            "multiclass_gray_jpg",
            "multiclass_color_jpg"
        ]

    def _apply_filters_to_patches(self, df_meta, patient_id, mask_col):
        if df_meta is None or df_meta.empty:
            return []

        filtered_rows = []

        for filt in self.filters:
            filt_safe = filt.replace("+", "_")
            filt_dir = os.path.join(self.save_root, f"{patient_id}_{mask_col}", "filtered", filt_safe)
            os.makedirs(filt_dir, exist_ok=True)

            for _, row_patch in df_meta.iterrows():
                patch_img_path = row_patch.get("image_patch_path", None)
                if not isinstance(patch_img_path, str) or not os.path.exists(patch_img_path):
                    continue

                patch_img = cv2.imread(patch_img_path, cv2.IMREAD_GRAYSCALE)
                if patch_img is None:
                    continue

                filtered_patch = apply_filter_chain(patch_img, filt)

                out_name = os.path.basename(patch_img_path)
                out_patch_path = os.path.join(filt_dir, out_name)
                cv2.imwrite(out_patch_path, filtered_patch)

                row_copy = row_patch.to_dict()
                row_copy["patient_id"] = patient_id
                row_copy["mask_col"] = mask_col
                row_copy["filter_name"] = filt
                row_copy["filtered_patch_path"] = out_patch_path
                filtered_rows.append(row_copy)

        return filtered_rows

    def run(self):
        from colab.src.image_utils.vertebra_component_extractor import VertebraComponentExtractor

        df = pd.read_csv(self.dataset_csv)

        if self.truncate is not None:
            start, stop = self.truncate
            df = df.iloc[start:stop]

        failed_cases = []
        all_centroids_metadata = []
        all_filtered_metadata = []

        mask_columns = self._get_mask_columns()

        for _, row in df.iterrows():
            patient_id = str(row["patient_id"])
            img_path = self._resolve_path(row.get("radiograph_path", None))

            if not img_path or not os.path.exists(img_path):
                failed_cases.append({
                    "patient_id": patient_id,
                    "reason": "radiograph_not_found",
                    "radiograph_path": img_path
                })
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed_cases.append({
                    "patient_id": patient_id,
                    "reason": "radiograph_load_failed",
                    "radiograph_path": img_path
                })
                continue

            for mask_col in mask_columns:
                mask_path = self._resolve_path(row.get(mask_col, None))
                if not mask_path or not os.path.exists(mask_path):
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    failed_cases.append({
                        "patient_id": patient_id,
                        "reason": "mask_load_failed",
                        "mask_col": mask_col,
                        "mask_path": mask_path
                    })
                    continue

                try:
                    patch_dir = os.path.join(self.save_root, f"{patient_id}_{mask_col}")
                    os.makedirs(patch_dir, exist_ok=True)

                    extractor = VertebraComponentExtractor(
                        image=img,
                        local_mask=mask,
                        min_area=150,
                        pad_x=20,
                        pad_y=15,
                        save_dir=patch_dir
                    )

                    extractor.run()
                    df_meta = extractor.save_patches_with_metadata(
                        sample_id=f"{patient_id}_{mask_col}"
                    )

                    if hasattr(extractor, "select_best_patches"):
                        df_meta = extractor.select_best_patches(df_meta)

                    if df_meta is None or df_meta.empty:
                        failed_cases.append({
                            "patient_id": patient_id,
                            "reason": "empty_metadata",
                            "mask_col": mask_col,
                            "mask_path": mask_path
                        })
                        continue

                    df_meta["patient_id"] = patient_id
                    df_meta["mask_col"] = mask_col
                    df_meta["mask_path"] = mask_path
                    df_meta["radiograph_path"] = img_path

                    centroids_csv = os.path.join(
                        self.centroid_curve_dir,
                        f"{patient_id}_{mask_col}_centroids_metadata.csv"
                    )
                    df_meta.to_csv(centroids_csv, index=False)

                    all_centroids_metadata.append(df_meta)

                    filtered_rows = self._apply_filters_to_patches(df_meta, patient_id, mask_col)
                    if filtered_rows:
                        all_filtered_metadata.append(pd.DataFrame(filtered_rows))

                except Exception as e:
                    failed_cases.append({
                        "patient_id": patient_id,
                        "reason": str(e),
                        "mask_col": mask_col,
                        "mask_path": mask_path
                    })

        if all_centroids_metadata:
            df_centroids = pd.concat(all_centroids_metadata, ignore_index=True)
            df_centroids.to_csv(
                os.path.join(self.save_root, "all_centroids_metadata.csv"),
                index=False
            )

        if all_filtered_metadata:
            df_filtered = pd.concat(all_filtered_metadata, ignore_index=True)
            df_filtered.to_csv(
                os.path.join(self.save_root, "all_filtered_patches_metadata.csv"),
                index=False
            )

        if failed_cases:
            pd.DataFrame(failed_cases).to_csv(
                os.path.join(self.save_root, "failed_cases.csv"),
                index=False
            )

        print("Pipeline finalizado.")