import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class PatchDTO:
    patch_id: str
    patient_id: str
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    bbox: Tuple[int, int, int, int] = None
    centroid_x: float = None
    centroid_y: float = None
    method: str = None

@dataclass
class PatchPathDTO:
    patch_id: str
    patient_id: str
    image_path: str
    mask_path: Optional[str]
    bbox: Tuple[int, int, int, int]
    centroid_x: float
    centroid_y: float
    method: str

class PatchDTOBuilder:
    def __init__(self, save_root: Optional[str] = None):
        self.save_root = save_root

    def _ensure_dirs(self):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio si quieres guardar en disco.")
        os.makedirs(self.save_root, exist_ok=True)
        img_dir = os.path.join(self.save_root, "patch_images")
        mask_dir = os.path.join(self.save_root, "patch_masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        return img_dir, mask_dir

    def build_patch_dtos_in_memory(self, patient_id: str, image: np.ndarray, mask: Optional[np.ndarray], boxes: List[dict], method: str, add_overlay: bool = False) -> List[PatchDTO]:
        dtos = []
        for item in boxes:
            idx = item["vertebra_idx"]
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            x1, y1, x2, y2 = item["bbox"]
            patch_id = f"{patient_id}_patch_{idx:02d}"
            patch_img = image[y1:y2, x1:x2].copy()
            patch_mask = mask[y1:y2, x1:x2].copy() if mask is not None else None
            overlay = None
            if add_overlay and patch_mask is not None:
                # Simple overlay: grayscale + mask in red
                overlay = np.stack([patch_img]*3, axis=-1)
                overlay[patch_mask > 0] = [255, 0, 0]
            dtos.append(
                PatchDTO(
                    patch_id=patch_id,
                    patient_id=patient_id,
                    image=patch_img,
                    mask=patch_mask,
                    overlay=overlay,
                    bbox=(x1, y1, x2, y2),
                    centroid_x=cx,
                    centroid_y=cy,
                    method=method
                )
            )
        return dtos

    def build_patch_dtos_on_disk(self, patient_id: str, image: np.ndarray, mask: Optional[np.ndarray], boxes: List[dict], method: str) -> List[PatchPathDTO]:
        img_dir, mask_dir = self._ensure_dirs()
        dtos = []
        for item in boxes:
            idx = item["vertebra_idx"]
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            x1, y1, x2, y2 = item["bbox"]
            patch_id = f"{patient_id}_patch_{idx:02d}"
            patch_img = image[y1:y2, x1:x2]
            patch_mask = mask[y1:y2, x1:x2] if mask is not None else None
            image_path = os.path.join(img_dir, f"{patch_id}.png")
            cv2.imwrite(image_path, patch_img)
            mask_path = None
            if patch_mask is not None:
                mask_path = os.path.join(mask_dir, f"{patch_id}_mask.png")
                cv2.imwrite(mask_path, patch_mask)
            dtos.append(
                PatchPathDTO(
                    patch_id=patch_id,
                    patient_id=patient_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    bbox=(x1, y1, x2, y2),
                    centroid_x=cx,
                    centroid_y=cy,
                    method=method
                )
            )
        return dtos
