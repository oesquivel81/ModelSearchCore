from pathlib import Path
import pandas as pd
import math
from PIL import Image
import matplotlib.pyplot as plt

def build_patch_images_paths_from_csv(config: dict) -> list[str]:
    """
    Construye rutas con este formato:
    {tda_root}/{patient_id}/{filter_name}_var-{use_variance}_mode-{variance_mode}_pk-{patch_size}_st-{stride}_vk-{variance_kernel}/patch_images
    Además:
    - imprime las rutas
    - opcionalmente muestra un grid de imágenes de ejemplo
    Todos los parámetros se extraen del diccionario de configuración.
    """
    # Extraer parámetros principales
    tda_root = config.get("tda_root")
    patient_id = config.get("patient_id")
    csv_path = config.get("csv_path")
    # Parámetros de visualización y control
    plot_cfg = config.get("patch_images_plot_config", {})
    unique_only = plot_cfg.get("unique_only", True)
    show_plot = plot_cfg.get("show_plot", False)
    images_per_folder = plot_cfg.get("images_per_folder", 4)
    max_folders_to_plot = plot_cfg.get("max_folders_to_plot", 6)

    if not all([tda_root, patient_id, csv_path]):
        raise ValueError("Faltan tda_root, patient_id o csv_path en la configuración.")

    df = pd.read_csv(csv_path)

    required_cols = [
        "filter_name",
        "use_variance",
        "variance_mode",
        "patch_size",
        "stride",
        "variance_kernel",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    def build_folder_name(row: pd.Series) -> str:
        return (
            f"{row['filter_name']}"
            f"_var-{row['use_variance']}"
            f"_mode-{row['variance_mode']}"
            f"_pk-{row['patch_size']}"
            f"_st-{row['stride']}"
            f"_vk-{row['variance_kernel']}"
        )

    base = Path(tda_root) / patient_id
    paths = []

    for _, row in df.iterrows():
        folder_name = build_folder_name(row)
        full_path = base / folder_name / "patch_images"
        paths.append(str(full_path))

    if unique_only:
        paths = list(dict.fromkeys(paths))

    print("Rutas construidas:")
    for p in paths:
        exists_flag = "✅" if Path(p).exists() else "❌"
        print(f"{exists_flag} {p}")

    if show_plot:
        _plot_patch_images_grid(
            patch_images_dirs=paths,
            images_per_folder=images_per_folder,
            max_folders=max_folders_to_plot,
        )

    return paths

def _plot_patch_images_grid(
    patch_images_dirs: list[str],
    images_per_folder: int = 4,
    max_folders: int = 6,
) -> None:
    """
    Muestra un grid con imágenes tomadas de las carpetas patch_images.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    selected = []

    for folder_str in patch_images_dirs[:max_folders]:
        folder = Path(folder_str)
        if not folder.exists() or not folder.is_dir():
            continue

        imgs = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        )[:images_per_folder]

        for img_path in imgs:
            selected.append((folder.parent.name, img_path))

    if not selected:
        print("No se encontraron imágenes para mostrar en el grid.")
        return

    n = len(selected)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax, (folder_name, img_path) in zip(axes, selected):
        try:
            img = Image.open(img_path)
            ax.imshow(img, cmap="gray" if img.mode in {"L", "I"} else None)
            ax.set_title(f"{folder_name}\n{img_path.name}", fontsize=8)
            ax.axis("off")
        except Exception as e:
            ax.set_title(f"Error: {img_path.name}", fontsize=8)
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
            ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
