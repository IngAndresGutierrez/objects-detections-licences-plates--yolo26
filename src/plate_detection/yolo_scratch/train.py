"""
YOLO26n entrenado DESDE CERO: arquitectura definida en yolo26n.yaml,
pesos inicializados aleatoriamente (sin COCO).

Uso (desde cualquier directorio):
    poetry run train-scratch --imgsz 320
    poetry run train-scratch --imgsz 416
"""

import argparse

from plate_detection.common.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    train(
        approach="yolo_scratch",
        model_source="yolo26n.yaml",
        pretrained=False,
        image_size=args.imgsz,
        name=args.name,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
