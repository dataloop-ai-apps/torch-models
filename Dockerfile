FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.4 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.4
