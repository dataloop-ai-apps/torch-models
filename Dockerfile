FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip
RUN pip install --user 'torch==2.0.1' 'torchvision==0.15.2' 'torchaudio==2.4.0'

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.4 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.4
