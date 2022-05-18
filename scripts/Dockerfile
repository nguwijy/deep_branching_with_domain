ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.02-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install --no-cache-dir sympy
RUN pip install --no-cache-dir ray
WORKDIR /deep_branching_with_domain

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install tzdata -y
ENV TZ="Asia/Singapore"

ENV CUDA_LAUNCH_BLOCKING=1
