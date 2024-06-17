# YOLOX Object Detection Model Adapter

## Introduction

This repo is a model integration
between [PyTorch DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) object
detection model and [Dataloop](https://dataloop.ai/).

DeepLab is a series of semantic segmentation models designed for high-performance scene understanding tasks.
By leveraging techniques such as atrous convolution and spatial pyramid pooling, DeepLab achieves state-of-the-art
performance in semantic segmentation while maintaining efficiency. This integration aims to provide a seamless workflow
for utilizing DeepLab models within the Dataloop platform.

## Requirements

* An account in the [Dataloop platform](https://console.dataloop.ai/)
* dtlpy
* torch>=1.8
* torchvision>0.9
* imbalanced-learn
* imblearn
* imgaug
* scikit-image>0.20.0
* scikit-learn

## Installation

To install the package and create the YOLOv5 model adapter, you will need
a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and
a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. The dataset should
have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory)
containing its training and validation subsets.

## Cloning

For instruction how to clone the pretrained model for prediction
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting)

## Training and Fine-tuning

For fine tuning on a custom dataset,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset)

### Editing the configuration

To edit configurations via the platform, go to the YOLOX page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

* ```epochs```: Number of epochs to train the model (default: ```100```)
* ```batch_size```: Batch size to be used during the training (default: ```16```)
* ```input_size```: Size of the input images for training (default: ```256```)
* ```augmentation```:  Boolean indicating whether to apply data augmentation during training (default: ```true```)
* ```conf_threshold```: Confidence threshold for predictions  (default ```0.8```)
* ```patience_epochs```: Number of epochs with no improvement after which training will be stopped (default ```10```)
* ```early_stopping```: Boolean indicating whether to apply early stopping by using ```patience_epochs``` (
  default ```true```)


