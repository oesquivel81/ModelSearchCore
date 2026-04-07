import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VertebraPatchDatasetV2(Dataset):
    def __init__(self, metadata_df, builder, model_type="baseline"):
        self.df = metadata_df.reset_index(drop=True)
        self.builder = builder
        self.model_type = model_type

    def __len__(self):
        return len(self.df)

    def _build_label(self, row):
        """
        Baseline inicial:
        usa el orden anatómico por centroid_y / component_idx.
        """
        cls = int(row["component_idx"])
        return min(cls, 12)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch = cv2.imread(row["image_patch_path"], cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(row["image_patch_path"])

        y = self._build_label(row)

        if self.model_type == "baseline":
            x = self.builder.build_baseline(patch)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        elif self.model_type == "variance_input":
            x = self.builder.build_variance_input(patch)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        elif self.model_type == "variance_branch":
            x_img, x_var = self.builder.build_variance_branch(patch)
            return {
                "image": torch.tensor(x_img, dtype=torch.float32),
                "variance": torch.tensor(x_var, dtype=torch.float32),
                "label": torch.tensor(y, dtype=torch.long)
            }

        else:
            raise ValueError(f"model_type no soportado: {self.model_type}")


class VertebraSubpatchDatasetFlexible(Dataset):
    def __init__(self, metadata_df, split, builder, model_type="baseline",
                 num_classes=13):
        self.df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
        self.builder = builder
        self.model_type = model_type
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def _build_label(self, row):
        cls = int(row["vertebra_idx"])
        return min(cls, self.num_classes - 1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch = cv2.imread(row["subpatch_img_path"], cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(row["subpatch_img_path"])

        y = self._build_label(row)

        if self.model_type == "baseline":
            x = self.builder.build_baseline(patch)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        elif self.model_type == "variance_input":
            x = self.builder.build_variance_input(patch)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        elif self.model_type == "variance_branch":
            x_img, x_var = self.builder.build_variance_branch(patch)
            return {
                "image": torch.tensor(x_img, dtype=torch.float32),
                "variance": torch.tensor(x_var, dtype=torch.float32),
                "label": torch.tensor(y, dtype=torch.long)
            }

        else:
            raise ValueError(f"model_type no soportado: {self.model_type}")
