# Detección de matrículas: YOLO desde cero, YOLO preentrenado y YOLOE

Detección de matrículas (una sola clase, `plate`) sobre un subconjunto del
dataset *Car License Plate* (imágenes `Cars*.png`). Se comparan tres enfoques:

| # | Enfoque | Carpeta | Qué es |
|---|---|---|---|
| 1 | **YOLO desde cero** | [`yolo_scratch/`](src/plate_detection/yolo_scratch/) | YOLO26n definido por `yolo26n.yaml`, pesos aleatorios, entrenado con datos de cov |
| 2 | **YOLO preentrenado** | [`yolo_pretrained/`](src/plate_detection/yolo_pretrained/) | YOLO26n con pesos COCO (`yolo26n.pt`) y *fine-tuning* entrenado con datos de cov |
| 3 | **YOLOE** | [`yoloe/`](src/plate_detection/yoloe/) | Detector de vocabulario abierto: la clase se indica con texto, sin entrenar |

## Estructura

```
.
├── README.md
├── pyproject.toml / poetry.lock  # dependencias (Poetry, Python 3.12, torch solo CPU)
├── poetry.toml                   # el entorno virtual se crea en .venv/
├── Dockerfile / docker-compose.yml / .dockerignore
├── Makefile                      # atajos: make prepare, make evaluate, make compare...
├── src/plate_detection/
│   ├── common/                   # código compartido por los tres enfoques
│   │   ├── dataset.py            # rutas, particiones, lectura de etiquetas
│   │   ├── training.py           # hiperparámetros y bucle de entrenamiento (1 y 2)
│   │   ├── inference.py          # inferencia guardando todas las cajas y puntuaciones
│   │   ├── metrics.py            # IoU, emparejamiento, P/R/F1, AP50
│   │   ├── visualization.py      # cajas, comparación original | predicción, galería
│   │   └── report.py             # guarda CSV, labels, imágenes, métricas y galería
│   ├── data_prep/prepare_dataset.py  # copia imágenes por partición, valida etiquetas, dataset.yaml
│   ├── yolo_scratch/train.py     # 1. entrenamiento desde cero
│   ├── yolo_pretrained/train.py  # 2. fine-tuning desde COCO
│   ├── yolo_eval/evaluate_test.py    # evaluación en test de 1 y 2
│   ├── yoloe/open_vocab.py       # 3. YOLOE: selección en val y evaluación en test
│   └── compare.py                # tabla comparativa final en test
├── data/
│   ├── voc/                      # dataset original (imágenes + anotaciones VOC, no usadas)
│   └── yolo/
│       ├── dataset.yaml
│       ├── splits/{train,val,test}.txt   # lista de imágenes de cada partición
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/      # anotación manual, formato YOLO
├── weights/                      # (no versionado) pesos y encoder de YOLOE, descargados solos
└── runs/
    ├── yolo_scratch/
    │   ├── train/img320/  train/img416/   # curvas, results.csv, args.yaml, weights/best.pt
    │   └── test/img320/                   # evaluación del run seleccionado
    ├── yolo_pretrained/
    │   ├── train/img416/
    │   └── test/img416/
    ├── yoloe/yoloe-26s-seg/
    │   ├── val_selection/                 # rejilla prompt × umbral en val
    │   └── test/
    ├── comparison.md / comparison.csv
    └── _archive/                          # smoke tests y un run interrumpido (no usados)
```

## Instalación

Hay dos opciones. En ambas se usa **solo CPU** y **Python 3.12**. Los resultados
de este README se obtuvieron en macOS (Apple M5) y se reproducen igual en Docker.

No hay que descargar nada a mano: `yolo26n.pt`, `yoloe-26s-seg.pt` y el encoder
de texto de YOLOE (`mobileclip2_b.ts`, 242 MB) se descargan solos en `weights/`,
que está en `.gitignore`.

### Opción A: Poetry (local)

Requiere [Poetry](https://python-poetry.org/docs/#installation) ≥ 2.0 y Python 3.12.

```bash
poetry env use python3.12   # solo la primera vez, si python3.12 no es el predeterminado
poetry install              # crea .venv/ e instala dependencias y comandos
```

### Opción B: Docker

Requiere Docker con Compose. `data/`, `weights/` y `runs/` se montan como
volúmenes, así que los resultados quedan en tu carpeta local.

```bash
docker compose build
```

Dentro de Docker no hay GPU ni MPS (en Mac). Todo corre en CPU, igual que los
experimentos originales.

## Ejecución completa

Cada paso es un comando del proyecto y funciona desde cualquier directorio.
Con Poetry se lanza con `poetry run <comando>`; con Docker, con
`docker compose run --rm app <comando>`.

```bash
poetry run prepare-dataset                 # 0. preparar y validar el dataset
poetry run train-scratch --imgsz 320       # 1. YOLO desde cero (~2 min en CPU)
poetry run train-scratch --imgsz 416
poetry run train-pretrained --imgsz 416    # 2. YOLO preentrenado (~3 min en CPU)
poetry run evaluate-test yolo_scratch      #    evaluar en test (1)
poetry run evaluate-test yolo_pretrained   #    evaluar en test (2)
poetry run yoloe                           # 3. YOLOE: val -> test
poetry run compare                         #    tabla comparativa
```

Ejemplo con Docker:

```bash
docker compose run --rm app evaluate-test yolo_pretrained
```

El `Makefile` agrupa estos pasos (`make prepare`, `make train-scratch`,
`make train-pretrained`, `make evaluate`, `make yoloe`, `make compare`,
`make all`). Con `RUN=docker` se ejecutan en el contenedor, por ejemplo
`make compare RUN=docker`.

---

## 1. YOLO desde cero

### Preparación de los datos

El dataset original (`data/voc/`) tiene 433 imágenes con anotaciones VOC. Se
usó un subconjunto de 55:

1. **Selección manual.** Se recorrieron las imágenes `Cars0`–`Cars81` en orden
   numérico y se descartaron las que no convencían. El reparto entre
   particiones también sigue ese orden, de modo que no se mezclan imágenes
   entre particiones:

   | Partición | Imágenes | Cajas | Rango | Lista |
   |---|---|---|---|---|
   | train | 35 | 39 | Cars0–Cars43 | [`data/yolo/splits/train.txt`](data/yolo/splits/train.txt) |
   | val | 15 | 17 | Cars44–Cars68 | [`data/yolo/splits/val.txt`](data/yolo/splits/val.txt) |
   | test | 5 | 7 | Cars70–Cars81 | [`data/yolo/splits/test.txt`](data/yolo/splits/test.txt) |

2. **Anotación manual.** Las matrículas se anotaron a mano en formato YOLO
   (`clase cx cy w h`, normalizados), con una única clase `0: plate`. Las
   anotaciones VOC originales **no se usan**: las cajas se dibujaron de nuevo.
   Están en `data/yolo/labels/<partición>/`.

3. **Script** [`data_prep/prepare_dataset.py`](src/plate_detection/data_prep/prepare_dataset.py).
   Copia cada imagen de `data/voc/images/` a su partición según las listas,
   comprueba que cada imagen tenga su etiqueta y que esta sea válida (5
   columnas, clase 0, coordenadas en [0, 1], ninguna imagen en dos
   particiones) y genera `dataset.yaml`. Con `--check` solo valida.

### Configuración del dataset

[`data/yolo/dataset.yaml`](data/yolo/dataset.yaml). No lleva `path`, así que
Ultralytics toma la carpeta del yaml como raíz y el repo funciona en cualquier
ubicación:

```yaml
train: images/train
val: images/val
test: images/test
names:
  0: plate
```

### Parámetros de entrenamiento

Están definidos en [`common/training.py`](src/plate_detection/common/training.py) y son **los mismos
para el enfoque 1 y el 2**. Lo único que cambia es el modelo de partida. La
lista completa de cada run está en su `args.yaml`.

| Parámetro | Valor |
|---|---|
| Modelo | `yolo26n.yaml` (arquitectura YOLO26n, 2.4 M parámetros) |
| `pretrained` | `False` (pesos aleatorios) |
| `imgsz` | 320 y 416 (dos runs) |
| `epochs` | 50 |
| `batch` | 4 |
| `optimizer` | `auto` (Ultralytics elige según el número de iteraciones) |
| `lr0` / `lrf` | 0.01 / 0.01 |
| `momentum` / `weight_decay` | 0.937 / 0.0005 |
| `warmup_epochs` | 3 |
| Aumentos | mosaic 1.0 (se desactiva en las últimas 10 epochs), fliplr 0.5, hsv (0.015, 0.7, 0.4), translate 0.1, scale 0.5 |
| `patience` | 100 (sin parada temprana efectiva) |
| `seed` / `deterministic` | 42 / `True` |
| `device` | `cpu` |
| Versión | Ultralytics 8.4.172, PyTorch 2.8.0 |

### Pesos finales, registros y curvas

El run final se elige **solo con validación**: el de mayor mAP50-95 en val. Lo
hace `src/plate_detection/yolo_eval/evaluate_test.py`. Dentro del run, `best.pt` es la epoch con
mejor *fitness* en val según Ultralytics.

| Run | Mejor epoch | val mAP50 | val mAP50-95 | |
|---|---|---|---|---|
| **img320** | 41 | 0.029 | 0.0065 | **seleccionado** |
| img416 | 49 | 0.012 | 0.0019 | |

- **Pesos finales:** [`runs/yolo_scratch/train/img320/weights/best.pt`](runs/yolo_scratch/train/img320/weights/best.pt)
  (los `best.pt` de los runs finales están versionados; `last.pt`, no).
- **Registro por epoch:** `results.csv` (pérdidas de train/val, precisión,
  recall, mAP50, mAP50-95 y learning rate).
- **Curvas:** `results.png` (evolución de pérdidas y métricas), `BoxPR_curve.png`,
  `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` y las matrices de confusión.
- **Lotes de ejemplo:** `train_batch*.jpg` (con los aumentos aplicados) y
  `val_batch*_{labels,pred}.jpg`.
- **Log de consola:** los entrenamientos que se lancen ahora guardan también
  `train.log` dentro del run. Los runs actuales se entrenaron antes de añadirlo,
  así que su registro es `results.csv`.

![Curvas de entrenamiento YOLO desde cero, img320](runs/yolo_scratch/train/img320/results.png)

**Conclusión:** con 35 imágenes de entrenamiento y 50 epochs, el modelo desde
cero no llega a aprender la tarea. En val apenas pasa de 0.03 de mAP50 y en
test no detecta nada por encima de 0.25. Las imágenes sin detecciones de la
galería salen de aquí.

## 2. YOLO preentrenado (fine-tuning)

Usa los mismos datos y parámetros que el enfoque 1, pero parte de `yolo26n.pt`
(COCO) con `pretrained=True` e `imgsz=416`. El run final es
[`runs/yolo_pretrained/train/img416/`](runs/yolo_pretrained/train/img416/):
mejor epoch 49, val mAP50 0.913, val mAP50-95 0.553. Sus pesos están en
`weights/best.pt`.

`runs/_archive/exp02_pretrained_img416_interrumpido` es un primer intento que
se cortó en la epoch 2. No se usa.

## 3. YOLOE (vocabulario abierto)

Usa `yoloe-26s-seg.pt` **sin entrenar**. La clase se indica con texto:

```python
from ultralytics import YOLOE
model = YOLOE("weights/yoloe-26s-seg.pt")
model.set_classes(["number plate"])   # embedding de texto con MobileCLIP
model.predict(image, conf=0.02, imgsz=640)
```

**Protocolo** (todas las decisiones se toman con val y se fijan antes de evaluar test):

1. Para cada prompt (`license plate`, `car license plate`, `number plate`,
   `vehicle registration plate`, `plate`, `car`) se calcula el AP50 en val y
   se elige el mejor: **`number plate`** (AP50 0.861).
2. Con ese prompt se elige en val el umbral que maximiza F1: **0.02** (P 0.73,
   R 0.94, F1 0.82). YOLOE da puntuaciones bajas a esta clase, por eso la rejilla
   empieza en 0.01.
3. Prompt y umbral se aplican **una sola vez** a test.

La rejilla completa está en `runs/yoloe/yoloe-26s-seg/val_selection/val_grid.csv`
y la configuración elegida en `selection.json`.

---

## Resultados en test

Cada evaluación en test guarda, en `runs/<enfoque>/test/...`:

- `predictions.csv`: todas las cajas con confianza ≥ 0.01 y su puntuación, si
  pasan el umbral (`kept`), si son TP o FP y su IoU con el GT.
- `predictions/labels/*.txt`: las cajas que pasan el umbral, en formato YOLO
  más la confianza.
- `predictions/*.png`: imágenes anotadas (verde = acierto, rojo = falso
  positivo, cian fino = ground truth).
- `comparisons/*.png`: imagen original **sin marcas** a la izquierda y la
  predicción a la derecha.
- `index.html`: galería agrupada en incorrectas, sin detecciones y correctas.
- `summary.csv` (una fila por imagen) y `metrics.json`.

Una predicción cuenta como acierto con IoU ≥ 0.5 (emparejamiento greedy por
confianza). AP50 se calcula con interpolación de 101 puntos sobre todas las
cajas guardadas (`poetry run compare`):

| Enfoque | Configuración | Umbral | TP | FP | FN | Precisión | Recall | F1 | AP50 |
|---|---|---|---|---|---|---|---|---|---|
| YOLO desde cero | img320, best.pt | 0.25 | 0 | 0 | 7 | 0.00 | 0.00 | 0.00 | 0.007 |
| YOLO preentrenado | img416, best.pt | 0.25 | 6 | 5 | 1 | 0.55 | 0.86 | 0.67 | 0.812 |
| YOLOE | yoloe-26s-seg, «number plate» | 0.02 | 7 | 5 | 0 | 0.58 | 1.00 | 0.74 | 1.000 |

Notas:

- En los dos YOLO el umbral 0.25 es el valor por defecto de Ultralytics y se
  fijó de antemano, sin ajustarlo. Como referencia, el mAP oficial de
  Ultralytics en test es 0.874 (mAP50) y 0.588 (mAP50-95) para el preentrenado,
  y 0.009 / 0.003 para el modelo desde cero. Está en `metrics.json`.
- El test tiene **5 imágenes y 7 matrículas**: un solo caso cambia mucho las
  métricas, así que las diferencias entre el enfoque 2 y el 3 no son concluyentes.
- **Cars70** tiene una matrícula sin anotar (el SUV de la izquierda). Los dos
  modelos que detectan algo la marcan y cuenta como falso positivo.
