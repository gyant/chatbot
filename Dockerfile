FROM python:3.11

RUN apt update
RUN apt install -y gcc

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install transformers sentencepiece gradio accelerate bitsandbytes pyyaml
