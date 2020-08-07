FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install tensorflow-compression
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev rsync vim git awscli

COPY . /thesis-compression
WORKDIR /thesis-compression
