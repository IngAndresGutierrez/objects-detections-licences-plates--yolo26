"""
Guarda los resultados de una partición evaluada a un umbral fijo:
cajas y puntuaciones (CSV y formato YOLO), imágenes anotadas,
comparaciones original | predicción, resumen, métricas y galería.
"""

import csv
import json

import cv2

from common.metrics import (
    IOU_MATCH_THRESHOLD,
    average_precision_50,
    classify,
    counts_at,
    filter_predictions,
    match_predictions,
)
from common.visualization import create_comparison, draw_predictions, write_gallery


def save_raw_predictions(records, csv_path, threshold=None, prompt=""):
    """
    Todas las cajas guardadas con su puntuación.
    Si se pasa umbral, se indica cuáles se conservan y si son TP/FP.
    """

    rows = []

    for record in records:
        results = {}

        if threshold is not None:
            kept = filter_predictions(record, threshold)
            matches, _ = match_predictions(kept, record["ground_truth"])
            results = {id(p): m for p, m in zip(kept, matches)}

        if not record["predictions"]:
            rows.append({"image": record["path"].name, "prompt": prompt})

        for prediction in sorted(record["predictions"],
                                 key=lambda p: p["confidence"], reverse=True):
            x1, y1, x2, y2 = prediction["box"]
            row = {
                "image": record["path"].name,
                "prompt": prompt,
                "confidence": round(prediction["confidence"], 4),
                "xmin": round(x1, 1), "ymin": round(y1, 1),
                "xmax": round(x2, 1), "ymax": round(y2, 1),
            }

            if threshold is not None and id(prediction) in results:
                match = results[id(prediction)]
                row["kept"] = 1
                row["result"] = "TP" if match is not None else "FP"
                row["iou_gt"] = round(match, 4) if match is not None else ""
            elif threshold is not None:
                row["kept"] = 0

            rows.append(row)

    fieldnames = ["image", "prompt", "confidence", "xmin", "ymin", "xmax", "ymax"]
    if threshold is not None:
        fieldnames += ["kept", "result", "iou_gt"]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def save_report(records, threshold, output_dir, title,
                label="plate", prompt="", extra_metrics=None, intro=""):

    predictions_dir = output_dir / "predictions"
    labels_dir = predictions_dir / "labels"
    comparisons_dir = output_dir / "comparisons"

    for directory in [predictions_dir, labels_dir, comparisons_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    save_raw_predictions(records, output_dir / "predictions.csv", threshold, prompt)

    summaries = []

    for record in records:
        image = record["image"]
        height, width = image.shape[:2]
        predictions = filter_predictions(record, threshold)
        matches, false_negatives = match_predictions(predictions, record["ground_truth"])

        true_positives = sum(m is not None for m in matches)
        false_positives = len(predictions) - true_positives

        summary = {
            "image": record["path"].name,
            "status": classify(len(predictions), false_positives, false_negatives),
            "ground_truth": len(record["ground_truth"]),
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
            f"{summary['image']}: {summary['status']} "
            f"(GT={summary['ground_truth']} TP={true_positives} "
            f"FP={false_positives} FN={false_negatives})"
        )

        # Predicciones en formato YOLO + confianza
        with open(labels_dir / f"{record['path'].stem}.txt", "w") as label_file:
            for prediction in predictions:
                x1, y1, x2, y2 = prediction["box"]
                label_file.write(
                    f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f} "
                    f"{prediction['confidence']:.4f}\n"
                )

        annotated = draw_predictions(image, predictions, matches,
                                     record["ground_truth"], label)
        cv2.imwrite(str(predictions_dir / record["path"].name), annotated)

        comparison = create_comparison(image, annotated, summary, threshold)
        cv2.imwrite(str(comparisons_dir / record["path"].name), comparison)

    with open(output_dir / "summary.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    counts = counts_at(records, threshold)

    metrics = {
        **(extra_metrics or {}),
        "confidence_threshold": threshold,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "test_images": len(records),
        "test_ap50": round(average_precision_50(records), 4),
        "test_tp": counts["tp"],
        "test_fp": counts["fp"],
        "test_fn": counts["fn"],
        "test_precision": round(counts["precision"], 4),
        "test_recall": round(counts["recall"], 4),
        "test_f1": round(counts["f1"], 4),
        "images_correctas": sum(s["status"] == "correcta" for s in summaries),
        "images_incorrectas": sum(s["status"] == "incorrecta" for s in summaries),
        "images_sin_detecciones": sum(s["status"] == "sin_detecciones" for s in summaries),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )

    gallery_metrics = {
        k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
        for k, v in metrics.items()
    }
    write_gallery(output_dir, title, summaries, gallery_metrics, threshold, intro)

    return metrics
