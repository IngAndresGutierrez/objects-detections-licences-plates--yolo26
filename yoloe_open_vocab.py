"""
Detección de matrículas con YOLOE (vocabulario abierto, prompts de texto).

Protocolo:
  1. En VALIDACIÓN se prueban varios prompts y se elige el de mayor AP50.
  2. Con ese prompt, también en validación, se elige el umbral de
     confianza que maximiza F1 (IoU >= 0.5).
  3. Prompt y umbral quedan FIJOS y se aplican una sola vez sobre TEST.

Uso: python yoloe_open_vocab.py [yoloe-26s-seg.pt]
"""

from pathlib import Path
import argparse
import csv
import html
import json

import cv2
import numpy as np
from ultralytics import YOLOE

from evaluate_test import (
    IMAGE_EXTENSIONS,
    IOU_MATCH_THRESHOLD,
    classify,
    create_comparison,
    draw_predictions,
    load_ground_truth,
    match_predictions,
    write_gallery,
)


DEFAULT_MODEL = "yoloe-26s-seg.pt"

VAL_IMAGES = Path("data/yolo/images/val")
VAL_LABELS = Path("data/yolo/labels/val")
TEST_IMAGES = Path("data/yolo/images/test")
TEST_LABELS = Path("data/yolo/labels/test")

PROMPTS = [
    "license plate",
    "car license plate",
    "number plate",
    "vehicle registration plate",
    "plate",
    "car",
]

# Umbral mínimo con el que se guardan TODAS las cajas; los umbrales
# candidatos se aplican después sobre estas puntuaciones.
MIN_CONFIDENCE = 0.01
# YOLOE puntúa bajo esta clase, por eso la rejilla empieza en 0.01
THRESHOLDS = [0.01, 0.02, 0.03, 0.04] + [round(t, 2) for t in np.arange(0.05, 0.95, 0.05)]
IMAGE_SIZE = 640


def list_images(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_prompt(model, prompt, images, labels_dir):
    """
    Inferencia con un prompt a confianza mínima.
    Devuelve, por imagen, la imagen, el GT y todas las cajas con puntuación.
    """

    model.set_classes([prompt])

    records = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        result = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=MIN_CONFIDENCE,
            device="cpu",
            verbose=False
        )[0]

        records.append({
            "path": image_path,
            "image": image,
            "ground_truth": load_ground_truth(image_path, width, height, labels_dir),
            "predictions": [
                {
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0]),
                }
                for box in result.boxes
            ],
        })

    return records


def filter_predictions(record, threshold):
    return [p for p in record["predictions"] if p["confidence"] >= threshold]


def counts_at(records, threshold):
    tp = fp = fn = 0

    for record in records:
        predictions = filter_predictions(record, threshold)
        matches, false_negatives = match_predictions(predictions, record["ground_truth"])

        hits = sum(m is not None for m in matches)
        tp += hits
        fp += len(predictions) - hits
        fn += false_negatives

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def average_precision_50(records):
    """
    AP a IoU 0.5 (interpolación de 101 puntos, estilo COCO)
    usando todas las cajas guardadas.
    """

    scored = []
    total_gt = 0

    for record in records:
        matches, _ = match_predictions(record["predictions"], record["ground_truth"])
        total_gt += len(record["ground_truth"])

        for prediction, match in zip(record["predictions"], matches):
            scored.append((prediction["confidence"], match is not None))

    if total_gt == 0 or not scored:
        return 0.0

    scored.sort(key=lambda item: item[0], reverse=True)
    hits = np.array([is_tp for _, is_tp in scored], dtype=float)

    tp_cumulative = np.cumsum(hits)
    fp_cumulative = np.cumsum(1 - hits)

    recall = tp_cumulative / total_gt
    precision = tp_cumulative / (tp_cumulative + fp_cumulative)

    # Envolvente de precisión monótona decreciente
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    ap = 0.0
    for recall_level in np.linspace(0, 1, 101):
        above = precision[recall >= recall_level]
        ap += above.max() if above.size else 0.0

    return ap / 101


def select_on_validation(model, output_dir):
    """
    Elige prompt (AP50) y umbral (F1) usando únicamente validación.
    """

    val_images = list_images(VAL_IMAGES)
    grid_rows = []
    prompt_scores = {}
    prompt_records = {}

    for prompt in PROMPTS:
        records = run_prompt(model, prompt, val_images, VAL_LABELS)
        prompt_records[prompt] = records
        prompt_scores[prompt] = average_precision_50(records)

        print(f"[val] '{prompt}': AP50={prompt_scores[prompt]:.4f}")

        for threshold in THRESHOLDS:
            counts = counts_at(records, threshold)
            grid_rows.append({
                "prompt": prompt,
                "threshold": threshold,
                "ap50": round(prompt_scores[prompt], 4),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in counts.items()},
            })

    best_prompt = max(PROMPTS, key=lambda p: prompt_scores[p])

    # Desempate: a igual F1 se prefiere el umbral más alto (menos FP)
    best_row = max(
        (row for row in grid_rows if row["prompt"] == best_prompt),
        key=lambda row: (row["f1"], row["threshold"])
    )

    with open(output_dir / "val_grid.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    save_raw_predictions(prompt_records[best_prompt], best_prompt,
                         output_dir / "val_raw_predictions.csv")

    return best_prompt, best_row["threshold"], prompt_scores, best_row


def save_raw_predictions(records, prompt, csv_path, threshold=None):
    """
    Todas las cajas (conf >= MIN_CONFIDENCE) con su puntuación.
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


def evaluate_test(model, prompt, threshold, output_dir):
    """
    Aplica prompt y umbral fijos (elegidos en validación) sobre test.
    """

    predictions_dir = output_dir / "predictions"
    labels_dir = predictions_dir / "labels"
    comparisons_dir = output_dir / "comparisons"

    for directory in [predictions_dir, labels_dir, comparisons_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    records = run_prompt(model, prompt, list_images(TEST_IMAGES), TEST_LABELS)

    save_raw_predictions(records, prompt, output_dir / "predictions.csv", threshold)

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
            f"[test] {summary['image']}: {summary['status']} "
            f"(GT={summary['ground_truth']} TP={true_positives} "
            f"FP={false_positives} FN={false_negatives})"
        )

        with open(labels_dir / f"{record['path'].stem}.txt", "w") as label_file:
            for prediction in predictions:
                x1, y1, x2, y2 = prediction["box"]
                label_file.write(
                    f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f} "
                    f"{prediction['confidence']:.4f}\n"
                )

        annotated = draw_predictions(image, predictions, matches,
                                     record["ground_truth"], prompt)
        cv2.imwrite(str(predictions_dir / record["path"].name), annotated)

        comparison = create_comparison(image, annotated, summary, threshold)
        cv2.imwrite(str(comparisons_dir / record["path"].name), comparison)

    with open(output_dir / "summary.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    counts = counts_at(records, threshold)

    return summaries, counts, average_precision_50(records)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    model_name = parser.parse_args().model

    output_dir = Path("runs/yoloe") / Path(model_name).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE(model_name)

    # -----------------------------------
    # 1-2. Selección en validación
    # -----------------------------------

    prompt, threshold, prompt_scores, val_row = select_on_validation(model, output_dir)

    print()
    print(f"Seleccionado en validación: prompt='{prompt}', umbral={threshold} "
          f"(val F1={val_row['f1']}, P={val_row['precision']}, R={val_row['recall']})")
    print()

    # -----------------------------------
    # 3. Test con la configuración fija
    # -----------------------------------

    summaries, counts, test_ap50 = evaluate_test(model, prompt, threshold, output_dir)

    metrics = {
        "model": model_name,
        "selection_split": "val",
        "prompt": prompt,
        "confidence_threshold": threshold,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "val_ap50_per_prompt": {p: round(s, 4) for p, s in prompt_scores.items()},
        "val_at_selected": {k: val_row[k] for k in
                            ["precision", "recall", "f1", "tp", "fp", "fn"]},
        "test_ap50": round(test_ap50, 4),
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

    intro = (
        f"<p>YOLOE con prompt de texto <b>&laquo;{html.escape(prompt)}&raquo;</b>. "
        f"Prompt (máx. AP50) y umbral (máx. F1) elegidos solo con validación "
        f"y fijados antes de evaluar test.</p>"
    )
    table_metrics = {k: v for k, v in metrics.items() if k != "val_ap50_per_prompt"}
    table_metrics.update({f"val AP50 '{p}'": round(s, 4) for p, s in prompt_scores.items()})

    write_gallery(output_dir, f"YOLOE {model_name}", summaries, table_metrics,
                  threshold, intro)

    print()
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    print(f"Galería: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
