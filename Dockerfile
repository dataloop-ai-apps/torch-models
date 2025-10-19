FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_opencv

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.7 -f ./Dockerfile  .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.7 bash
# docker push gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.7
