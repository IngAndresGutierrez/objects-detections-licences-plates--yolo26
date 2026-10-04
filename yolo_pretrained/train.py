from ultralytics import YOLO
import platform
import time
import torch


def main():
    # Configuración
    epochs = 50
    image_size = 416
    batch_size = 4
    seed = 42

    print("=" * 60)
    print("YOLO26n - fine-tuning desde pesos preentrenados (COCO)")
    print("=" * 60)

    print(f"Python hardware: {platform.machine()}")
    print(f"CPU: {platform.processor()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print("Device: CPU")
    print("Pretrained: True")
    print("=" * 60)

    # IMPORTANTE:
    # .pt = arquitectura + pesos preentrenados en COCO
    model = YOLO("yolo26n.pt")

    start_time = time.time()

    results = model.train(
        data="data/yolo/dataset.yaml",

        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,

        device="cpu",

        pretrained=True,

        seed=seed,
        deterministic=True,

        project="experiments",
        name="exp02_pretrained_img416",

        save=True,
        plots=True,
    )

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Tiempo total: {elapsed / 60:.2f} minutos")
    print("=" * 60)


if __name__ == "__main__":
    main()