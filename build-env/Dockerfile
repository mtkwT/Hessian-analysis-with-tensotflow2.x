FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

RUN pip install --upgrade pip

# Install package in python
WORKDIR /opt/backend
ADD requirements.txt /opt/backend/requirements.txt

RUN pip install pip --upgrade && \
    pip install -r requirements.txt