from pathlib import Path
from typing import List, Optional
import argparse
import csv
import html
import json

import cv2
import numpy as np
from ultralytics import YOLO


# Run completo de 50 epochs (exp02_pretrained_img416 se interrumpió en la epoch 2).
# Se puede evaluar otro experimento con: python evaluate_test.py exp01_img416
DEFAULT_EXPERIMENT = "exp02_pretrained_img416-2"
DATASET_YAML = "data/yolo/dataset.yaml"
TEST_IMAGES = Path("data/yolo/images/test")
TEST_LABELS = Path("data/yolo/labels/test")

CONFIDENCE_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5
IMAGE_SIZE = 416

# Colores BGR
COLOR_TP = (0, 200, 0)
COLOR_FP = (0, 0, 255)
COLOR_GT = (255, 200, 0)

STATUS_COLORS = {
    "correcta": (0, 160, 0),
    "incorrecta": (0, 0, 220),
    "sin_detecciones": (0, 140, 255),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_ground_truth(image_path, width, height):
    """
    Lee las etiquetas YOLO (clase cx cy w h normalizados)
    y las devuelve como cajas xyxy en píxeles.
    """

    label_path = TEST_LABELS / f"{image_path.stem}.txt"

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


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions(predictions, ground_truth):
    """
    Emparejamiento greedy por confianza descendente.
    Devuelve, por cada predicción, el IoU con su GT asignado
    (o None si es falso positivo) y el número de GT sin detectar.
    """

    order = sorted(
        range(len(predictions)),
        key=lambda i: predictions[i]["confidence"],
        reverse=True
    )

    matched_gt = set()
    matches: List[Optional[float]] = [None] * len(predictions)

    for i in order:
        best_iou = 0.0
        best_gt = None

        for j, gt_box in enumerate(ground_truth):
            if j in matched_gt:
                continue

            overlap = iou(predictions[i]["box"], gt_box)

            if overlap > best_iou:
                best_iou = overlap
                best_gt = j

        if best_gt is not None and best_iou >= IOU_MATCH_THRESHOLD:
            matched_gt.add(best_gt)
            matches[i] = best_iou

    false_negatives = len(ground_truth) - len(matched_gt)

    return matches, false_negatives


def draw_predictions(image, predictions, matches, ground_truth):
    """
    Dibuja el GT (fino, cian) y las predicciones con su confianza:
    verde = acierto (TP), rojo = falso positivo (FP).
    """

    annotated = image.copy()

    for gt_box in ground_truth:
        x1, y1, x2, y2 = map(int, gt_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_GT, 1)

    for prediction, match in zip(predictions, matches):
        x1, y1, x2, y2 = map(int, prediction["box"])
        color = COLOR_TP if match is not None else COLOR_FP
        text = f"plate {prediction['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = y1 - 4 if y1 - text_h - baseline - 4 >= 0 else y2 + text_h + 4

        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - baseline),
            (x1 + text_w + 4, text_y + baseline),
            color,
            -1
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated


def create_comparison(original, annotated, summary):
    """
    Original sin marcas a la izquierda y la inferencia a la derecha.
    Los títulos y el resumen van en bandas fuera de las imágenes
    para no tapar ningún píxel de la original.
    """

    height, width = original.shape[:2]
    separator = np.full((height, 8, 3), 255, dtype=np.uint8)

    body = np.hstack([original, separator, annotated])
    total_width = body.shape[1]

    header = np.full((36, total_width, 3), 40, dtype=np.uint8)
    cv2.putText(header, "ORIGINAL", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "PREDICCION", (width + 18, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    footer = np.full((36, total_width, 3), 0, dtype=np.uint8)
    footer[:] = STATUS_COLORS[summary["status"]]

    footer_text = (
        f"{summary['image']} | {summary['status'].upper()} | "
        f"GT={summary['ground_truth']} TP={summary['tp']} "
        f"FP={summary['fp']} FN={summary['fn']} | conf>={CONFIDENCE_THRESHOLD}"
    )
    cv2.putText(footer, footer_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, body, footer])


def classify(number_predictions, false_positives, false_negatives):
    if number_predictions == 0:
        return "sin_detecciones"

    if false_positives == 0 and false_negatives == 0:
        return "correcta"

    return "incorrecta"


def write_gallery(output_dir, experiment, summaries, metrics):
    """
    Galería HTML estática: una tarjeta por imagen, agrupadas por estado.
    """

    sections = []

    for status in ["incorrecta", "sin_detecciones", "correcta"]:
        items = [s for s in summaries if s["status"] == status]

        if not items:
            continue

        cards = "\n".join(
            f'<figure class="{status}">'
            f'<img src="comparisons/{html.escape(s["image"])}" loading="lazy">'
            f'<figcaption><b>{html.escape(s["image"])}</b> &middot; '
            f'GT {s["ground_truth"]} &middot; TP {s["tp"]} &middot; '
            f'FP {s["fp"]} &middot; FN {s["fn"]} &middot; '
            f'conf máx {s["max_confidence"] or "-"}</figcaption></figure>'
            for s in items
        )

        sections.append(
            f"<h2>{status.replace('_', ' ').capitalize()} ({len(items)})</h2>\n{cards}"
        )

    metric_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in metrics.items()
    )

    page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Evaluación test {experiment}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f4f4f4; }}
figure {{ background: #fff; margin: 0 0 20px; padding: 8px; border-left: 6px solid #999; }}
figure.correcta {{ border-color: #00a000; }}
figure.incorrecta {{ border-color: #dc0000; }}
figure.sin_detecciones {{ border-color: #ff8c00; }}
img {{ max-width: 100%; display: block; }}
table {{ border-collapse: collapse; background: #fff; }}
td {{ border: 1px solid #ccc; padding: 4px 10px; }}
</style>
</head>
<body>
<h1>Evaluación sobre test &mdash; {experiment}</h1>
<p>Izquierda: imagen original sin marcas. Derecha: inferencia
(conf &ge; {CONFIDENCE_THRESHOLD}). Verde = acierto (IoU &ge; {IOU_MATCH_THRESHOLD}),
rojo = falso positivo, cian fino = ground truth.</p>
<table>{metric_rows}</table>
{"".join(sections)}
</body>
</html>
"""

    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default=DEFAULT_EXPERIMENT)
    experiment = parser.parse_args().experiment

    model_path = f"runs/detect/experiments/{experiment}/weights/best.pt"

    output_dir = Path(f"runs/evaluate/{experiment}")
    predictions_dir = output_dir / "predictions"
    labels_dir = predictions_dir / "labels"
    comparisons_dir = output_dir / "comparisons"

    for directory in [predictions_dir, labels_dir, comparisons_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    test_images = sorted(
        path
        for path in TEST_IMAGES.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Modelo: {model_path}")
    print(f"Test images: {len(test_images)}")

    box_rows = []
    summaries = []

    # -----------------------------------
    # Inferencia
    # -----------------------------------

    for image_path in test_images:

        original = cv2.imread(str(image_path))
        height, width = original.shape[:2]

        result = model.predict(
            source=original,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device="cpu",
            verbose=False
        )[0]

        predictions = [
            {
                "box": box.xyxy[0].cpu().numpy().tolist(),
                "confidence": float(box.conf[0]),
            }
            for box in result.boxes
        ]

        ground_truth = load_ground_truth(image_path, width, height)
        matches, false_negatives = match_predictions(predictions, ground_truth)

        true_positives = sum(m is not None for m in matches)
        false_positives = len(predictions) - true_positives

        summary = {
            "image": image_path.name,
            "status": classify(len(predictions), false_positives, false_negatives),
            "ground_truth": len(ground_truth),
            "predictions": len(predictions),
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "max_confidence": (
                round(max(p["confidence"] for p in predictions), 4)
                if predictions else ""
            ),
        }
        summaries.append(summary)

        print(
            f"{image_path.name}: {summary['status']} "
            f"(GT={len(ground_truth)} TP={true_positives} "
            f"FP={false_positives} FN={false_negatives})"
        )

        # Predicciones en formato YOLO + confianza
        with open(labels_dir / f"{image_path.stem}.txt", "w") as label_file:
            for prediction in predictions:
                x1, y1, x2, y2 = prediction["box"]
                label_file.write(
                    f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f} "
                    f"{prediction['confidence']:.4f}\n"
                )

        if not predictions:
            box_rows.append({
                "image": image_path.name,
                "confidence": "",
                "xmin": "", "ymin": "", "xmax": "", "ymax": "",
                "iou_gt": "",
                "result": "",
            })

        for prediction, match in zip(predictions, matches):
            x1, y1, x2, y2 = prediction["box"]
            box_rows.append({
                "image": image_path.name,
                "confidence": round(prediction["confidence"], 4),
                "xmin": round(x1, 1), "ymin": round(y1, 1),
                "xmax": round(x2, 1), "ymax": round(y2, 1),
                "iou_gt": round(match, 4) if match is not None else "",
                "result": "TP" if match is not None else "FP",
            })

        annotated = draw_predictions(original, predictions, matches, ground_truth)
        cv2.imwrite(str(predictions_dir / image_path.name), annotated)

        comparison = create_comparison(original, annotated, summary)
        cv2.imwrite(str(comparisons_dir / image_path.name), comparison)

    # -----------------------------------
    # Métricas oficiales sobre el split test
    # -----------------------------------

    validation = model.val(
        data=DATASET_YAML,
        split="test",
        imgsz=IMAGE_SIZE,
        device="cpu",
        project=str(output_dir.resolve()),
        name="val_test",
        exist_ok=True,
        verbose=False
    )

    total_tp = sum(s["tp"] for s in summaries)
    total_fp = sum(s["fp"] for s in summaries)
    total_fn = sum(s["fn"] for s in summaries)

    metrics = {
        "model": model_path,
        "test_images": len(test_images),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision@conf": round(total_tp / (total_tp + total_fp), 4) if total_tp + total_fp else 0.0,
        "recall@conf": round(total_tp / (total_tp + total_fn), 4) if total_tp + total_fn else 0.0,
        "mAP50": round(float(validation.box.map50), 4),
        "mAP50-95": round(float(validation.box.map), 4),
        "images_correctas": sum(s["status"] == "correcta" for s in summaries),
        "images_incorrectas": sum(s["status"] == "incorrecta" for s in summaries),
        "images_sin_detecciones": sum(s["status"] == "sin_detecciones" for s in summaries),
    }

    # -----------------------------------
    # Guardar resultados
    # -----------------------------------

    with open(output_dir / "predictions.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(box_rows[0].keys()))
        writer.writeheader()
        writer.writerows(box_rows)

    with open(output_dir / "summary.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    write_gallery(output_dir, experiment, summaries, metrics)

    print()
    print("=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    print(f"Galería: {output_dir / 'index.html'}")
    print(f"Comparaciones: {comparisons_dir}")
    print(f"Predicciones: {predictions_dir}")


if __name__ == "__main__":
    main()
