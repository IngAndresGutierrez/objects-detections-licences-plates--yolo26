"""
Evaluación en TEST de los YOLO entrenados (desde cero o preentrenado).

El run final de cada enfoque se elige SOLO con validación: el de mayor
mAP50-95 en val durante el entrenamiento (results.csv). Su best.pt es,
a su vez, la epoch con mejor fitness en val según Ultralytics.

Uso (desde la raíz del repo):
    python -m yolo_eval.evaluate_test yolo_scratch
    python -m yolo_eval.evaluate_test yolo_pretrained
    python -m yolo_eval.evaluate_test yolo_scratch --run img416   # forzar un run
"""

import argparse
import csv
import shutil

import yaml
from ultralytics import YOLO

from common.dataset import DATASET_YAML, RUNS_DIR
from common.inference import run_model
from common.report import save_report

# Umbral por defecto de Ultralytics, fijado a priori (no se ajusta con test)
CONFIDENCE_THRESHOLD = 0.25


def best_val_scores(run_dir):
    rows = list(csv.DictReader(open(run_dir / "results.csv")))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))

    return {
        "epoch": int(best["epoch"]),
        "val_mAP50": float(best["metrics/mAP50(B)"]),
        "val_mAP50-95": float(best["metrics/mAP50-95(B)"]),
    }


def select_run(approach):
    candidates = {
        run_dir.name: best_val_scores(run_dir)
        for run_dir in sorted((RUNS_DIR / approach / "train").iterdir())
        if (run_dir / "weights" / "best.pt").exists()
    }

    selected = max(candidates, key=lambda name: candidates[name]["val_mAP50-95"])

    return selected, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("approach", choices=["yolo_scratch", "yolo_pretrained"])
    parser.add_argument("--run", default=None, help="por defecto, el mejor en val")
    args = parser.parse_args()

    selected, candidates = select_run(args.approach)
    run_name = args.run or selected

    run_dir = RUNS_DIR / args.approach / "train" / run_name
    model_path = run_dir / "weights" / "best.pt"
    train_args = yaml.safe_load(open(run_dir / "args.yaml"))
    image_size = train_args["imgsz"]

    output_dir = RUNS_DIR / args.approach / "test" / run_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"Candidatos (val): {candidates}")
    print(f"Run evaluado: {run_name} (imgsz={image_size})")

    model = YOLO(str(model_path))
    records = run_model(model, "test", image_size)

    # mAP oficial de Ultralytics sobre test, como referencia adicional
    validation = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=image_size,
        device="cpu",
        project=str(output_dir.resolve()),
        name="ultralytics_val",
        exist_ok=True,
        verbose=False
    )

    metrics = save_report(
        records,
        CONFIDENCE_THRESHOLD,
        output_dir,
        title=f"{args.approach} / {run_name}",
        extra_metrics={
            "approach": args.approach,
            "run": run_name,
            "model": str(model_path.relative_to(RUNS_DIR.parent)),
            "image_size": image_size,
            "selection_split": "val",
            "selection_criterion": "max val mAP50-95",
            "val_candidates": candidates,
            "ultralytics_test_mAP50": round(float(validation.box.map50), 4),
            "ultralytics_test_mAP50-95": round(float(validation.box.map), 4),
        },
    )

    print()
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    print(f"Galería: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
