"""
Entrenamiento común a YOLO desde cero y YOLO preentrenado.
Ambos usan exactamente los mismos hiperparámetros; solo cambian
el punto de partida (yaml vs .pt) y el tamaño de imagen.
"""

import logging
import platform
import time

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from common.dataset import DATASET_YAML, RUNS_DIR

# Hiperparámetros explícitos. El resto quedan en los valores por defecto
# de Ultralytics 8.4.172 y se registran en args.yaml de cada run.
TRAIN_PARAMS = {
    "epochs": 50,
    "batch": 4,
    "device": "cpu",
    "seed": 42,
    "deterministic": True,
    "optimizer": "auto",
    "patience": 100,
    "save": True,
    "plots": True,
}


def train(approach, model_source, pretrained, image_size, name=None, epochs=None):
    name = name or f"img{image_size}"
    params = {**TRAIN_PARAMS, "epochs": epochs or TRAIN_PARAMS["epochs"]}

    # Ultralytics antepone runs/detect a los project relativos
    project = (RUNS_DIR / approach / "train").resolve()
    run_dir = project / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copia del log de consola junto a los resultados del run
    log_handler = logging.FileHandler(run_dir / "train.log", mode="w")
    LOGGER.addHandler(log_handler)

    print("=" * 60)
    print(f"Enfoque: {approach}")
    print(f"Modelo: {model_source} (pretrained={pretrained})")
    print(f"Hardware: {platform.machine()} / PyTorch {torch.__version__}")
    print(f"Image size: {image_size}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 60)

    model = YOLO(model_source)

    start_time = time.time()

    model.train(
        data=str(DATASET_YAML),
        imgsz=image_size,
        pretrained=pretrained,
        project=str(project),
        name=name,
        exist_ok=True,
        **params,
    )

    elapsed = time.time() - start_time
    LOGGER.info(f"Tiempo total: {elapsed / 60:.2f} minutos")
    LOGGER.removeHandler(log_handler)

    print(f"Resultados: {run_dir}")
