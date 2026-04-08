# =============================
# Visualización de parches
# =============================
import matplotlib.pyplot as plt
import cv2

def show_patches(patch_dtos, max_cols=5, show_mask=True, show_overlay=False):
    """
    Visualiza una lista de PatchDTO o PatchPathDTO.
    Si es PatchDTO, usa los arrays en memoria.
    Si es PatchPathDTO, carga desde disco.
    Si show_overlay=True y el patch tiene overlay, lo muestra.
    """
    n = len(patch_dtos)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(3 * ncols, 3 * nrows))
    for i, patch in enumerate(patch_dtos):
        plt.subplot(nrows, ncols, i + 1)
        # Detecta tipo
        if hasattr(patch, "image") and patch.image is not None:
            img = patch.image
            mask = patch.mask if hasattr(patch, "mask") else None
            overlay = patch.overlay if hasattr(patch, "overlay") else None
        elif hasattr(patch, "image_path"):
            img = cv2.imread(patch.image_path, cv2.IMREAD_GRAYSCALE)
            mask = None
            overlay = None
            if patch.mask_path is not None:
                mask = cv2.imread(patch.mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            continue
        if show_overlay and overlay is not None:
            plt.imshow(overlay)
        elif show_mask and mask is not None:
            plt.imshow(img, cmap='gray')
            plt.imshow(mask, cmap='jet', alpha=0.4)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(f"{patch.patch_id}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
