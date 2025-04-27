FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

RUN pip install -r requirements.txt