from ultralytics import YOLO
import platform
import time
import torch


def main():
    # Configuración
    epochs = 5
    image_size = 320
    batch_size = 4
    seed = 42

    print("=" * 60)
    print("YOLO26n - entrenamiento desde cero")
    print("=" * 60)

    print(f"Python hardware: {platform.machine()}")
    print(f"CPU: {platform.processor()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print("Device: CPU")
    print("Pretrained: False")
    print("=" * 60)

    # IMPORTANTE:
    # .yaml = arquitectura
    # NO estamos cargando yolo26n.pt
    model = YOLO("yolo26n.yaml")

    start_time = time.time()

    results = model.train(
        data="data/yolo/dataset.yaml",

        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,

        device="cpu",

        pretrained=False,

        seed=seed,
        deterministic=True,

        project="experiments",
        name="smoke_test",

        save=True,
        plots=True,
    )

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Tiempo total: {elapsed / 60:.2f} minutos")
    print("=" * 60)


if __name__ == "__main__":
    main()