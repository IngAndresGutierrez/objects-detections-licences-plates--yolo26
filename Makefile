# Atajos. Con Docker: make docker-build y luego make <objetivo> RUN=docker
RUN ?= local

ifeq ($(RUN),docker)
EXEC = docker compose run --rm app
else
EXEC = poetry run
endif

.PHONY: install docker-build prepare train-scratch train-pretrained evaluate yoloe compare all

install:
	poetry install

docker-build:
	docker compose build

prepare:
	$(EXEC) prepare-dataset

train-scratch:
	$(EXEC) train-scratch --imgsz 320
	$(EXEC) train-scratch --imgsz 416

train-pretrained:
	$(EXEC) train-pretrained --imgsz 416

evaluate:
	$(EXEC) evaluate-test yolo_scratch
	$(EXEC) evaluate-test yolo_pretrained

yoloe:
	$(EXEC) yoloe

compare:
	$(EXEC) compare

all: prepare train-scratch train-pretrained evaluate yoloe compare
