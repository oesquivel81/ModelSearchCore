import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VertebraRegionDataset(Dataset):
    """
    PyTorch Dataset que sirve parches pre-extraídos de regiones vertebrales.

    Cada elemento es una VertebraRegion con patch_img y patch_mask.
    Los parches se redimensionan a patch_size y se normalizan a [0, 1].

    Args:
        regions: lista de VertebraRegion
        patch_size: (H, W) tamaño objetivo para resize
        binarize_mask: si True, binariza la máscara con umbral 127
    """

    def __init__(self, regions, patch_size=(128, 128), binarize_mask=True):
        self.regions = list(regions)
        self.patch_size = patch_size
        self.binarize_mask = binarize_mask

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        r = self.regions[idx]

        img = cv2.resize(r.patch_img, self.patch_size[::-1], interpolation=cv2.INTER_AREA)
        mask = cv2.resize(r.patch_mask, self.patch_size[::-1], interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0

        if self.binarize_mask:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
