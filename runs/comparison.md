| Enfoque | Configuración | Umbral | TP | FP | FN | Precisión | Recall | F1 | AP50 |
|---|---|---|---|---|---|---|---|---|---|
| yolo_scratch | img320, best.pt | 0.25 | 0 | 0 | 7 | 0.0 | 0.0 | 0.0 | 0.0068 |
| yolo_pretrained | img416, best.pt | 0.25 | 6 | 5 | 1 | 0.5455 | 0.8571 | 0.6667 | 0.812 |
| yoloe | yoloe-26s-seg.pt, prompt 'number plate' | 0.02 | 7 | 5 | 0 | 0.5833 | 1.0 | 0.7368 | 1.0 |
