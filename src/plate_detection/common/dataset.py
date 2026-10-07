"""
Rutas del proyecto y lectura del dataset YOLO (una sola clase: plate).
Las rutas son absolutas respecto a la raíz del repo para que los
scripts funcionen desde cualquier directorio. PROJECT_ROOT permite
cambiar la raíz (en Docker apunta a /app).
"""

import os
from pathlib import Path

# src/plate_detection/common/dataset.py -> raíz del repo
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))

DATA_DIR = ROOT / "data" / "yolo"
DATASET_YAML = DATA_DIR / "dataset.yaml"
SPLITS_DIR = DATA_DIR / "splits"
VOC_IMAGES = ROOT / "data" / "voc" / "images"

WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"

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
