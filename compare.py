"""
Tabla comparativa en test de los tres enfoques, a partir de sus metrics.json.

Uso (desde la raíz del repo, después de evaluar los tres):
    python compare.py
"""

import csv
import json

from common.dataset import RUNS_DIR

COLUMNS = [
    ("approach", "Enfoque"),
    ("config", "Configuración"),
    ("confidence_threshold", "Umbral"),
    ("test_tp", "TP"),
    ("test_fp", "FP"),
    ("test_fn", "FN"),
    ("test_precision", "Precisión"),
    ("test_recall", "Recall"),
    ("test_f1", "F1"),
    ("test_ap50", "AP50"),
]


def main():
    rows = []

    metrics_paths = (
        sorted(RUNS_DIR.glob("yolo_scratch/test/*/metrics.json"))
        + sorted(RUNS_DIR.glob("yolo_pretrained/test/*/metrics.json"))
        + sorted(RUNS_DIR.glob("yoloe/*/test/metrics.json"))
    )

    for metrics_path in metrics_paths:
        metrics = json.loads(metrics_path.read_text())

        if metrics["approach"] == "yoloe":
            config = f"{metrics['model']}, prompt '{metrics['prompt']}'"
        else:
            config = f"{metrics['run']}, best.pt"

        rows.append({**metrics, "config": config})

    header = [title for _, title in COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key, _ in COLUMNS) + " |")

    table = "\n".join(lines)
    print(table)

    (RUNS_DIR / "comparison.md").write_text(table + "\n")

    with open(RUNS_DIR / "comparison.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[key] for key, _ in COLUMNS])


if __name__ == "__main__":
    main()
