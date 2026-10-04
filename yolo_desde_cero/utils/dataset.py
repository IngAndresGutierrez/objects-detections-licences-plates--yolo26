"""
Rutas del proyecto y lectura del dataset YOLO (una sola clase: plate).
Las rutas son absolutas respecto a esta carpeta para que los scripts
funcionen desde cualquier directorio.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
DATASET_YAML = DATA_DIR / "dataset.yaml"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_YAML = ROOT / "model" / "yolo26n.yaml"
RUNS_DIR = ROOT / "runs"

# Imágenes originales del dataset Car License Plate (fuera de esta carpeta)
DEFAULT_SOURCE_IMAGES = ROOT.parent / "data" / "voc" / "images"

SPLITS = ("train", "val", "test")
CLASS_NAMES = ["plate"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def images_dir(split):
    return DATA_DIR / "images" / split


def labels_dir(split):
    return DATA_DIR / "labels" / split


def list_images(split):
    return sorted(
        path
        for path in images_dir(split).iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_split_list(split):
    """Nombres (sin extensión) de las imágenes de una partición."""

    path = SPLITS_DIR / f"{split}.txt"

    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_ground_truth(image_path, width, height, split):
    """
    Lee las etiquetas YOLO (clase cx cy w h normalizados)
    y las devuelve como cajas xyxy en píxeles.
    """

    label_path = labels_dir(split) / f"{image_path.stem}.txt"

    if not label_path.exists():
        return []

    boxes = []

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        _, cx, cy, w, h = map(float, parts)

        boxes.append([
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        ])

    return boxes
