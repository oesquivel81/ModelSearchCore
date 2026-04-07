import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

from .variance_patch_processor import VariancePatchProcessor


class VertebraPatchDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        processor: VariancePatchProcessor,
        model_type: str = "baseline",
        label_mode: str = "ordinal_13"
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.processor = processor
        self.model_type = model_type
        self.label_mode = label_mode

    def __len__(self):
        return len(self.df)

    def _build_label(self, row):
        """
        Baseline inicial:
        usa component_idx como clase 0..12.
        Filtra fuera si excede 12.
        """
        cls = int(row["component_idx"])
        if cls > 12:
            cls = 12
        return cls

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch = cv2.imread(row["image_patch_path"], cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(row["image_patch_path"])

        y = self._build_label(row)

        if self.model_type == "baseline":
            x = self.processor.build_baseline_tensor(patch)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        if self.model_type == "variance_input":
            x = self.processor.build_variance_input_tensor(patch)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        if self.model_type == "variance_branch":
            x_img, x_var = self.processor.build_variance_branch_tensor(patch)
            return {
                "image": torch.tensor(x_img, dtype=torch.float32),
                "variance": torch.tensor(x_var, dtype=torch.float32),
                "label": torch.tensor(y, dtype=torch.long)
            }

        raise ValueError(f"model_type no soportado: {self.model_type}")
