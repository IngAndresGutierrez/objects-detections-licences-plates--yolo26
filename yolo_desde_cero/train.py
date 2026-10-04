"""
YOLO26n entrenado DESDE CERO con nuestras imágenes: arquitectura
definida en model/yolo26n.yaml y pesos inicializados aleatoriamente
(sin pesos preentrenados).

Uso (desde esta carpeta):
    python train.py --imgsz 320
    python train.py --imgsz 416
"""

import argparse
import logging
import platform
import time

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from utils.dataset import DATASET_YAML, MODEL_YAML, RUNS_DIR

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
    "pretrained": False,
    "save": True,
    "plots": True,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or f"img{args.imgsz}"
    params = {**TRAIN_PARAMS, "epochs": args.epochs or TRAIN_PARAMS["epochs"]}

    # Ultralytics antepone runs/detect a los project relativos
    project = (RUNS_DIR / "train").resolve()
    run_dir = project / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copia del log de consola junto a los resultados del run
    log_handler = logging.FileHandler(run_dir / "train.log", mode="w")
    LOGGER.addHandler(log_handler)

    print("=" * 60)
    print(f"YOLO26n desde cero: {MODEL_YAML.name}")
    print(f"Hardware: {platform.machine()} / PyTorch {torch.__version__}")
    print(f"Image size: {args.imgsz}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 60)

    # El nombre del yaml (yolo26n) fija la escala "n" de la arquitectura
    model = YOLO(str(MODEL_YAML))

    start_time = time.time()

    model.train(
        data=str(DATASET_YAML),
        imgsz=args.imgsz,
        project=str(project),
        name=name,
        exist_ok=True,
        **params,
    )

    elapsed = time.time() - start_time
    LOGGER.info(f"Tiempo total: {elapsed / 60:.2f} minutos")
    LOGGER.removeHandler(log_handler)

    print(f"Resultados: {run_dir}")


if __name__ == "__main__":
    main()
