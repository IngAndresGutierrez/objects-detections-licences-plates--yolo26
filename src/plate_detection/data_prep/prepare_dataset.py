"""
Preparación del dataset YOLO a partir de las imágenes originales.

Flujo seguido en el proyecto:
  1. Selección MANUAL de 55 imágenes de data/voc/images (Cars0–Cars81,
     por orden numérico, descartando las que no convencían) y reparto
     por orden: train Cars0–43, val Cars44–68, test Cars70–81.
     El resultado está versionado en data/yolo/splits/{train,val,test}.txt.
  2. Anotación MANUAL de las matrículas (clase única "plate") en formato
     YOLO, en data/yolo/labels/<split>/. Las anotaciones VOC originales
     de data/voc/annotations NO se usan.
  3. Este script: copia las imágenes a cada partición, valida las
     etiquetas y genera dataset.yaml.

Uso (desde cualquier directorio):
    poetry run prepare-dataset           # copia, valida y escribe yaml
    poetry run prepare-dataset --check   # solo valida
"""

import argparse
import shutil

from plate_detection.common.dataset import (
    CLASS_NAMES,
    DATA_DIR,
    DATASET_YAML,
    SPLITS,
    VOC_IMAGES,
    images_dir,
    labels_dir,
    read_split_list,
)


def validate_label_file(path):
    """Devuelve (número de cajas, lista de errores) de un .txt YOLO."""

    errors = []
    boxes = 0

    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(f"{path.name}:{number} se esperaban 5 columnas")
            continue

        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(f"{path.name}:{number} clase {class_id} fuera de rango")

        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w == 0 or h == 0:
            errors.append(f"{path.name}:{number} coordenadas no normalizadas o caja vacía")

        boxes += 1

    return boxes, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="solo validar")
    args = parser.parse_args()

    errors = []
    seen = {}

    for split in SPLITS:
        stems = read_split_list(split)

        for stem in stems:
            if stem in seen:
                errors.append(f"{stem} aparece en {seen[stem]} y en {split}")
            seen[stem] = split

        if not args.check:
            images_dir(split).mkdir(parents=True, exist_ok=True)
            for stem in stems:
                shutil.copy2(VOC_IMAGES / f"{stem}.png", images_dir(split) / f"{stem}.png")

        total_boxes = 0
        for stem in stems:
            if not (images_dir(split) / f"{stem}.png").exists():
                errors.append(f"{split}/{stem}.png no existe")

            label_path = labels_dir(split) / f"{stem}.txt"
            if not label_path.exists():
                errors.append(f"{split}/{stem}.txt sin anotar")
                continue

            boxes, label_errors = validate_label_file(label_path)
            total_boxes += boxes
            errors += [f"{split}/{e}" for e in label_errors]

        # Imágenes en la carpeta que no están en la lista
        listed = set(stems)
        extra = [p.stem for p in images_dir(split).glob("*.png") if p.stem not in listed]
        errors += [f"{split}/{stem}.png no está en splits/{split}.txt" for stem in extra]

        print(f"{split:>5}: {len(stems):3d} imágenes, {total_boxes:3d} cajas")

    if not args.check:
        # Sin "path": Ultralytics usa la carpeta del yaml como raíz,
        # así el repo funciona en cualquier ubicación.
        names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
        DATASET_YAML.write_text(
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"names:\n{names}\n"
        )
        print(f"Escrito {DATASET_YAML.relative_to(DATA_DIR.parent.parent)}")

    if errors:
        print("\nERRORES:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("Dataset OK")


if __name__ == "__main__":
    main()
