build:
	docker build --tag pytorch:gpu .

build-cpu:
	docker build --tag pytorch:cpu -f Dockerfile.cpu .

run:
	docker run --rm -it --gpus all --mount type=bind,src=C:\code\torch-model-test,target=/app --workdir /app -p 7860:7860 pytorch:gpu /bin/bash

run-cpu:
	docker run --rm -it --mount type=bind,src=$(shell pwd),target=/app --workdir /app -p 7860:7860 pytorch:cpu /bin/bash
