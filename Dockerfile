FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2
USER root

RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN pip3 install --upgrade pip

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.3 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.3
