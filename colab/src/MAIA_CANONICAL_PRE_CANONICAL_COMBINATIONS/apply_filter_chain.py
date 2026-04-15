import cv2
import numpy as np

def apply_filter_chain(image, filter_chain):
    """
    Aplica una cadena de filtros separados por '+'.
    Soporta: 'none', 'gaussian', 'median', 'sobel', 'laplacian', 'bilateral', 'clahe', 'unsharp_mask', 'scharr'.
    """
    filters = filter_chain.lower().split("+")
    img = image.copy()
    for f in filters:
        if f == "none":
            continue
        elif f == "gaussian":
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif f == "median":
            img = cv2.medianBlur(img, 5)
        elif f == "sobel":
            img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
            img = cv2.convertScaleAbs(img)
        elif f == "laplacian":
            img = cv2.Laplacian(img, cv2.CV_64F)
            img = cv2.convertScaleAbs(img)
        elif f == "bilateral":
            img = cv2.bilateralFilter(img, 9, 75, 75)
        elif f == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
        elif f == "unsharp_mask":
            blur = cv2.GaussianBlur(img, (9, 9), 10.0)
            img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        elif f == "scharr":
            img = cv2.Scharr(img, cv2.CV_64F, 1, 0) + cv2.Scharr(img, cv2.CV_64F, 0, 1)
            img = cv2.convertScaleAbs(img)
        else:
            raise ValueError(f"Filtro no soportado: {f}")
    return img