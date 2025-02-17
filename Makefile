build:
	docker build --platform linux/amd64 --tag chatbot:latest .

build-cpu:
	docker build --tag chatbot:cpu -f Dockerfile.cpu .

run-windows:
	docker run --rm -it --gpus all --mount type=bind,src=$(shell cd),target=/app --workdir /app -p 7860:7860 chatbot:latest /bin/bash

run:
	docker run --rm -it --gpus all --mount type=bind,src=$(shell pwd),target=/app --workdir /app -p 7860:7860 chatbot:latest /bin/bash

run-cpu:
	docker run --rm -it --mount type=bind,src=$(shell pwd),target=/app --workdir /app -p 7860:7860 chatbot:cpu /bin/bash
