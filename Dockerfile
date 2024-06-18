FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:latest

USER 1000
ENV HOME=/tmp
RUN pip3 install --upgrade pip

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/piper/agent/gpu/torch-models:0.1.0 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/gpu/torch-models:0.1.0
