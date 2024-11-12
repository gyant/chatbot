FROM python:3.11

RUN apt update
RUN apt install -y gcc

COPY requirements.txt .

RUN pip3 install -r requirements.txt
