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

    def build_patch_dtos_in_memory(self, patient_id: str, image: np.ndarray, mask: Optional[np.ndarray], boxes: List[dict], method: str, add_overlay: bool = False, patch_size=None, stride=None) -> List[PatchDTO]:
        dtos = []
        for item in boxes:
            idx = item["vertebra_idx"]
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            x1, y1, x2, y2 = item["bbox"]
            # Si patch_size está definido, forzar el tamaño del parche
            if patch_size is not None:
                w, h = patch_size
                x2 = x1 + w
                y2 = y1 + h
            patch_id = f"{patient_id}_patch_{idx:02d}"
            patch_img = image[y1:y2, x1:x2].copy()
            patch_mask = mask[y1:y2, x1:x2].copy() if mask is not None else None
            overlay = None
            if add_overlay and patch_mask is not None:
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

    def build_patch_dtos_on_disk(self, patient_id: str, image: np.ndarray, mask: Optional[np.ndarray], boxes: List[dict], method: str, patch_size=None, stride=None) -> List[PatchPathDTO]:
        print(f"[TRACE] build_patch_dtos_on_disk llamado para patient_id={patient_id}, method={method}")
        # Crear subcarpetas por paciente y método (filtro)
        patient_dir = os.path.join(self.save_root, str(patient_id))
        method_dir = os.path.join(patient_dir, str(method))
        img_dir = os.path.join(method_dir, "patch_images")
        mask_dir = os.path.join(method_dir, "patch_masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        print(f"[TRACE] Directorios: img_dir={img_dir}, mask_dir={mask_dir}")
        print(f"[TRACE] Número de boxes recibidos: {len(boxes)}")
        dtos = []
        for item in boxes:
            print(f"[TRACE] Procesando box: {item}")
            idx = item["vertebra_idx"]
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            x1, y1, x2, y2 = item["bbox"]
            # Si patch_size está definido, forzar el tamaño del parche
            if patch_size is not None:
                w, h = patch_size
                x2 = x1 + w
                y2 = y1 + h
            patch_id = f"{patient_id}_patch_{idx:02d}"
            patch_img = image[y1:y2, x1:x2]
            patch_mask = mask[y1:y2, x1:x2] if mask is not None else None
            image_path = os.path.join(img_dir, f"{patch_id}.png")
            save_img = _prepare_patch_for_saving(patch_img)
            saved_img = cv2.imwrite(image_path, save_img)
            print(f"[TRACE] Guardando imagen: {image_path} - {'OK' if saved_img else 'FALLO'}")
            if not saved_img:
                raise IOError(f"cv2.imwrite falló para {image_path}")
            mask_path = None
            if patch_mask is not None:
                mask_path = os.path.join(mask_dir, f"{patch_id}_mask.png")
                save_mask = _prepare_patch_for_saving(patch_mask)
                saved_mask = cv2.imwrite(mask_path, save_mask)
                print(f"[TRACE] Guardando máscara: {mask_path} - {'OK' if saved_mask else 'FALLO'}")
                if not saved_mask:
                    raise IOError(f"cv2.imwrite falló para {mask_path}")
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
        print(f"[TRACE] Total de parches generados: {len(dtos)}")
        return dtos

# === Utilidades para guardar imágenes de parches con cualquier shape/canales ===
def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    min_v = np.min(img)
    max_v = np.max(img)
    if max_v - min_v < 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)
    img = (img - min_v) / (max_v - min_v)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

def _prepare_patch_for_saving(patch_img: np.ndarray) -> np.ndarray:
    patch_img = np.asarray(patch_img)
    if patch_img.ndim == 2:
        return _to_uint8(patch_img)
    if patch_img.ndim == 3:
        h, w, c = patch_img.shape
        if c == 1:
            return _to_uint8(patch_img[..., 0])
        if c == 2:
            ch0 = _to_uint8(patch_img[..., 0])
            ch1 = _to_uint8(patch_img[..., 1])
            ch2 = np.zeros((h, w), dtype=np.uint8)
            return np.stack([ch0, ch1, ch2], axis=-1)
        if c in (3, 4):
            return _to_uint8(patch_img)
    raise ValueError(
        f"No se puede guardar patch con shape={patch_img.shape}, dtype={patch_img.dtype}")
