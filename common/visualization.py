"""
Visualizaciones: cajas sobre la imagen, comparación original | predicción
y galería HTML.
"""

import html

import cv2
import numpy as np

from common.metrics import IOU_MATCH_THRESHOLD

# Colores BGR
COLOR_TP = (0, 200, 0)
COLOR_FP = (0, 0, 255)
COLOR_GT = (255, 200, 0)

STATUS_COLORS = {
    "correcta": (0, 160, 0),
    "incorrecta": (0, 0, 220),
    "sin_detecciones": (0, 140, 255),
}


def draw_predictions(image, predictions, matches, ground_truth, label="plate"):
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
        text = f"{label} {prediction['confidence']:.2f}"

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


def create_comparison(original, annotated, summary, confidence_threshold):
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
        f"FP={summary['fp']} FN={summary['fn']} | conf>={confidence_threshold}"
    )
    cv2.putText(footer, footer_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, body, footer])


def write_gallery(output_dir, title, summaries, metrics, confidence_threshold, intro=""):
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
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in metrics.items()
    )

    page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Evaluación test {html.escape(title)}</title>
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
<h1>Evaluación sobre test &mdash; {html.escape(title)}</h1>
<p>Izquierda: imagen original sin marcas. Derecha: inferencia
(conf &ge; {confidence_threshold}). Verde = acierto (IoU &ge; {IOU_MATCH_THRESHOLD}),
rojo = falso positivo, cian fino = ground truth.</p>
{intro}
<table>{metric_rows}</table>
{"".join(sections)}
</body>
</html>
"""

    (output_dir / "index.html").write_text(page, encoding="utf-8")
