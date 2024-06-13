FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:latest

USER 1000
ENV HOME=/tmp
RUN pip3 install --upgrade pip
RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.86.8-py3-none-any.whl --upgrade --user
RUN pip3 install https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.86.8.0-py3-none-any.whl --upgrade --user

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build -t gcr.io/viewo-g/modelmgmt/resnet:0.0.8 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/modelmgmt/resnet:0.0.8
