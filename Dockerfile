FROM python:3.11

RUN apt update
RUN apt install -y gcc

RUN pip install torch torchvision torchaudio
RUN pip install transformers sentencepiece gradio
