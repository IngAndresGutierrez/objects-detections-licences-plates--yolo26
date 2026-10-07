"""
Métricas comunes a los tres enfoques.

Un "record" es el resultado de una imagen:
    {"path", "image", "ground_truth": [xyxy], "predictions": [{"box", "confidence"}]}
"""

from typing import List, Optional

import numpy as np

IOU_MATCH_THRESHOLD = 0.5


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


def filter_predictions(record, threshold):
    return [p for p in record["predictions"] if p["confidence"] >= threshold]


def classify(number_predictions, false_positives, false_negatives):
    if number_predictions == 0:
        return "sin_detecciones"

    if false_positives == 0 and false_negatives == 0:
        return "correcta"

    return "incorrecta"


def counts_at(records, threshold):
    """TP/FP/FN, precisión, recall y F1 a un umbral de confianza."""

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
