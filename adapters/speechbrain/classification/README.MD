# SpeechBrain Encoder Decoder Model Adapter

## Introduction

This repo is a model integration between SpeechBrain for audio classification and [Dataloop](https://dataloop.ai/).

SpeechBrain is an open-source PyTorch toolkit that accelerates Conversational AI development, i.e., the technology behind speech assistants, chatbots, and large language models.
## Model Available

- [Spoken Language Identification Model](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)

## Requirements

- dtlpy
- torch>=1.8.0
- torchvision>=0.9
- speechbrain 
- torchaudio 
- soundfile
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the SpeechBrain model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform.

## Deployment

After installing the pretrained model, it is necessary to deploy it, so it can be used
for prediction.

### Editing the configuration

To edit configurations, go to the model page in the Model Management section on the platform and edit the displayed JSON file or use the SDK to edit the model configuration. Basic configuration include:

- `conf_thresh`: confidence threshold for which a language is added to the predicted label.

## Sources and Further Reading

- [SpeechBrain Documentation](https://speechbrain.readthedocs.io/en/latest/)

## Acknowledgements

The original models paper and codebase can be found here:
- SpeechBrain paper on [arXiv](https://arxiv.org/abs/2106.04624) and codebase on [GitHub](https://github.com/speechbrain/speechbrain).

We appreciate their efforts in advancing the field and making their work accessible to the broader community.