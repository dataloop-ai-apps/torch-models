FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:latest
USER root

RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN pip3 install --upgrade pip

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.2 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.2

