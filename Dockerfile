FROM python:3.11

RUN apt update
RUN apt install -y gcc

RUN mkdir /app

COPY requirements.txt /app
COPY main.py /app

WORKDIR /app
RUN pip3 install -r requirements.txt
