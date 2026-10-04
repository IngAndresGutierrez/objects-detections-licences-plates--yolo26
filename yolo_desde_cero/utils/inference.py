"""
Inferencia sobre una partición guardando todas las cajas y puntuaciones.
"""

import cv2

from utils.dataset import list_images, load_ground_truth

# Umbral mínimo con el que se guardan TODAS las cajas y sus puntuaciones;
# el umbral de trabajo se aplica después sobre ellas.
MIN_CONFIDENCE = 0.01


def run_model(model, split, image_size):
    records = []

    for image_path in list_images(split):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        result = model.predict(
            source=image,
            imgsz=image_size,
            conf=MIN_CONFIDENCE,
            device="cpu",
            verbose=False
        )[0]

        records.append({
            "path": image_path,
            "image": image,
            "ground_truth": load_ground_truth(image_path, width, height, split),
            "predictions": [
                {
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0]),
                }
                for box in result.boxes
            ],
        })

    return records
