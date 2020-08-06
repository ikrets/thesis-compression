FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN pip install tensorflow-compression
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev rsync vim git

COPY . /thesis-compression
WORKDIR /thesis-compression
