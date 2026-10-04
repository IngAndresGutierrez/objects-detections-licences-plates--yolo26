"""
Detección de matrículas con YOLOE (vocabulario abierto, prompts de texto),
sin ningún entrenamiento sobre nuestro dataset.

Protocolo:
  1. En VALIDACIÓN se prueban varios prompts y se elige el de mayor AP50.
  2. Con ese prompt, también en validación, se elige el umbral de
     confianza que maximiza F1 (IoU >= 0.5).
  3. Prompt y umbral quedan FIJOS y se aplican una sola vez sobre TEST.

Uso (desde la raíz del repo):
    python -m yoloe.open_vocab                    # yoloe-26s-seg.pt
    python -m yoloe.open_vocab yoloe-26m-seg.pt
"""

import argparse
import csv
import html
import json
import shutil

from ultralytics import YOLOE

from common.dataset import RUNS_DIR, WEIGHTS_DIR
from common.inference import run_model
from common.metrics import average_precision_50, counts_at
from common.report import save_raw_predictions, save_report

DEFAULT_MODEL = "yoloe-26s-seg.pt"
IMAGE_SIZE = 640

PROMPTS = [
    "license plate",
    "car license plate",
    "number plate",
    "vehicle registration plate",
    "plate",
    "car",
]

# YOLOE puntúa bajo esta clase, por eso la rejilla empieza en 0.01
THRESHOLDS = [0.01, 0.02, 0.03, 0.04] + [round(0.05 * i, 2) for i in range(1, 19)]


def run_prompt(model, prompt, split):
    model.set_classes([prompt])

    return run_model(model, split, IMAGE_SIZE)


def select_on_validation(model, output_dir):
    """
    Elige prompt (AP50) y umbral (F1) usando únicamente validación.
    """

    grid_rows = []
    prompt_scores = {}
    prompt_records = {}

    for prompt in PROMPTS:
        records = run_prompt(model, prompt, "val")
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

    save_raw_predictions(prompt_records[best_prompt],
                         output_dir / "val_raw_predictions.csv",
                         prompt=best_prompt)

    return best_prompt, best_row["threshold"], prompt_scores, best_row


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    model_name = parser.parse_args().model

    base_dir = RUNS_DIR / "yoloe" / model_name.replace(".pt", "")
    if base_dir.exists():
        shutil.rmtree(base_dir)

    selection_dir = base_dir / "val_selection"
    test_dir = base_dir / "test"
    selection_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Se descarga automáticamente en weights/ si no existe.
    # El encoder de texto (mobileclip2_b.ts) se descarga en el directorio actual.
    model = YOLOE(str(WEIGHTS_DIR / model_name))

    # -----------------------------------
    # 1-2. Selección en validación
    # -----------------------------------

    prompt, threshold, prompt_scores, val_row = select_on_validation(model, selection_dir)

    selection = {
        "model": model_name,
        "selection_split": "val",
        "prompt_criterion": "max val AP50",
        "threshold_criterion": "max val F1 (IoU >= 0.5)",
        "prompt": prompt,
        "confidence_threshold": threshold,
        "val_ap50_per_prompt": {p: round(s, 4) for p, s in prompt_scores.items()},
        "val_at_selected": {k: val_row[k] for k in
                            ["precision", "recall", "f1", "tp", "fp", "fn"]},
    }
    (selection_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False)
    )

    print()
    print(f"Seleccionado en validación: prompt='{prompt}', umbral={threshold} "
          f"(val F1={val_row['f1']}, P={val_row['precision']}, R={val_row['recall']})")
    print()

    # -----------------------------------
    # 3. Test con la configuración fija
    # -----------------------------------

    records = run_prompt(model, prompt, "test")

    intro = (
        f"<p>YOLOE con prompt de texto <b>&laquo;{html.escape(prompt)}&raquo;</b>. "
        f"Prompt (máx. AP50) y umbral (máx. F1) elegidos solo con validación "
        f"y fijados antes de evaluar test.</p>"
    )

    metrics = save_report(
        records,
        threshold,
        test_dir,
        title=f"YOLOE {model_name}",
        label=prompt,
        prompt=prompt,
        extra_metrics={
            "approach": "yoloe",
            "model": model_name,
            "image_size": IMAGE_SIZE,
            **{k: v for k, v in selection.items()
               if k not in ("model", "confidence_threshold")},
        },
        intro=intro,
    )

    print()
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    print(f"Galería: {test_dir / 'index.html'}")


if __name__ == "__main__":
    main()
