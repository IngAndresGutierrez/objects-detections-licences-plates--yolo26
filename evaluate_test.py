from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "runs/detect/experiments/exp02_pretrained_img416/weights/best.pt"
TEST_IMAGES = Path("data/yolo/images/test")

OUTPUT_DIR = Path("runs/evaluate/exp02_pretrained_img416")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
COMPARISONS_DIR = OUTPUT_DIR / "comparisons"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 416


def create_comparison(original, prediction, filename):
    """
    Coloca la imagen original a la izquierda
    y la predicción a la derecha.
    """

    # Ambas deben tener la misma altura
    height = max(original.shape[0], prediction.shape[0])

    def resize_to_height(image, target_height):
        h, w = image.shape[:2]

        if h == target_height:
            return image

        scale = target_height / h

        new_width = int(w * scale)

        return cv2.resize(
            image,
            (new_width, target_height)
        )

    original = resize_to_height(original, height)
    prediction = resize_to_height(prediction, height)

    # Añadir títulos
    cv2.putText(
        original,
        "ORIGINAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        prediction,
        "PREDICTION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    comparison = np.hstack([
        original,
        prediction
    ])

    output_path = COMPARISONS_DIR / filename

    cv2.imwrite(
        str(output_path),
        comparison
    )


def main():

    # -----------------------------------
    # Crear carpetas
    # -----------------------------------

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    COMPARISONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------
    # Cargar modelo entrenado
    # -----------------------------------

    model = YOLO(MODEL_PATH)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    test_images = [
        path
        for path in TEST_IMAGES.iterdir()
        if path.suffix.lower() in image_extensions
    ]

    print(
        f"Test images: {len(test_images)}"
    )

    csv_rows = []

    # -----------------------------------
    # Inferencia
    # -----------------------------------

    for image_path in sorted(test_images):

        print(
            f"Processing: {image_path.name}"
        )

        original = cv2.imread(
            str(image_path)
        )

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # -----------------------------------
        # Imagen con cajas de Ultralytics
        # -----------------------------------

        annotated = result.plot()

        if len(result.boxes) == 0:
            cv2.putText(
                annotated,
                f"SIN DETECCIONES (conf >= {CONFIDENCE_THRESHOLD})",
                (20, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        prediction_path = (
            PREDICTIONS_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(prediction_path),
            annotated
        )

        # -----------------------------------
        # Guardar información de detecciones
        # -----------------------------------

        number_detections = len(
            result.boxes
        )

        if number_detections == 0:

            csv_rows.append({
                "image": image_path.name,
                "detections": 0,
                "confidence": "",
                "xmin": "",
                "ymin": "",
                "xmax": "",
                "ymax": ""
            })

        else:

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                xmin, ymin, xmax, ymax = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                csv_rows.append({
                    "image": image_path.name,
                    "detections": number_detections,
                    "confidence": confidence,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })

        # -----------------------------------
        # Crear comparación
        # -----------------------------------

        create_comparison(
            original.copy(),
            annotated.copy(),
            image_path.name
        )

    # -----------------------------------
    # Guardar CSV
    # -----------------------------------

    csv_path = (
        OUTPUT_DIR /
        "predictions.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as csv_file:

        fieldnames = [
            "image",
            "detections",
            "confidence",
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    print()
    print("=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)
    print(
        f"Predictions: {PREDICTIONS_DIR}"
    )
    print(
        f"Comparisons: {COMPARISONS_DIR}"
    )
    print(
        f"CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()