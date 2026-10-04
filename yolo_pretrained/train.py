"""
YOLO26n con FINE-TUNING desde pesos preentrenados en COCO (yolo26n.pt).
Mismos hiperparámetros que yolo_scratch; solo cambia el punto de partida.

Uso (desde la raíz del repo):
    python -m yolo_pretrained.train --imgsz 416
"""

import argparse

from common.dataset import WEIGHTS_DIR
from common.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    train(
        approach="yolo_pretrained",
        # Se descarga automáticamente en weights/ si no existe
        model_source=str(WEIGHTS_DIR / "yolo26n.pt"),
        pretrained=True,
        image_size=args.imgsz,
        name=args.name,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
