FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_opencv

RUN apt update && apt install -y curl

RUN ${DL_PYTHON_EXECUTABLE} -m pip install --upgrade pip
RUN ${DL_PYTHON_EXECUTABLE} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY /requirements.txt .

RUN ${DL_PYTHON_EXECUTABLE} -m pip install  -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.10 -f ./Dockerfile  .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.10 bash
# docker push gcr.io/viewo-g/piper/agent/runner/apps/torch-models:0.1.10