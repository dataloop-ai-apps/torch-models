# TIMM Model Adapter

## Introduction

This repo is a model integration between TIMM models for classification [TIMM Models](https://huggingface.co/timm) and [Dataloop](https://dataloop.ai/).

PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

## Models available

Various models and their variants can be used. To switch models, edit the model name in the configuration after installation from the Marketplace. The default models are:  trained or finetuned onImageNet1k (1k at the end of the model).

- EfficientNet (default: 'efficientnetv2_rw_s.ra2_in1k')
- HRNet (default: 'hrnet_w18.ms_aug_in1k')
- MobileNet (default: 'mobilenetv3_large_100.ra_in1k')

Other variants from these models can be found [here](https://huggingface.co/models?pipeline_tag=image-classification&library=timm) and used if trained or fine-tuned on ImageNet1k can also be used.

## Requirements

- dtlpy
- torch>=1.8.0
- torchvision>=0.9
- imbalanced-learn
- imgaug
- timm
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the TIMM model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory), tags or you can use DQL filter to have training and validation subsets.

## Training and Fine-tuning

For finetuning on a custom dataset, follow the instructions [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

### Editing the configuration

To edit configurations, go to the TIMM model page in the Model Management section on the platform and edit the displayed JSON file or use the SDK to edit the model configuration. Basic configurations include:

- `model_name`: the name of the model from [here](https://huggingface.co/models?pipeline_tag=image-classification&library=timm) (default: see 'Models available')
- `weights_filename`:  name of the model weights file (default: "model.pth")
- `batch_size`: batch size to be used during the training (default: 16)
- `num_epochs`: number of epochs to train the model (default: 10)

Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more information.

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

## Sources and Further Reading

- [EfficientNet Documentation](https://huggingface.co/docs/transformers/en/model_doc/efficientnet)
- [HRNet Documentation](https://huggingface.co/docs/timm/en/models/hrnet)
- [MobileNetV3 Documentation](https://pytorch.org/vision/main/models/mobilenetv3.html)

## Acknowledgements

The original models paper and codebase can be found here:
- EfficientNet paper on [arXiv](https://arxiv.org/abs/1905.11946) and codebase on [GitHub](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py)
- HRNet paper on [arXiv](https://arxiv.org/abs/1908.07919) and codebase on [GitHub](https://github.com/HRNet/HRNet-Image-Classification)
- MobileNet paper on [arXiv](https://arxiv.org/pdf/1905.02244) and codebase on [GitHub](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py).

We appreciate their efforts in advancing the field and making their work accessible to the broader community.