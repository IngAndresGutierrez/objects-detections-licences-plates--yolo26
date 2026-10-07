# Imagen solo CPU. data/, weights/ y runs/ se montan como volúmenes
# (ver docker-compose.yml), así que no se copian a la imagen.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PROJECT_ROOT=/app \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# git: dependencia clip desde GitHub; libgl1/libglib2.0-0: opencv-python
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only main

CMD ["compare"]
