# ResNet Model Adapter

## Introduction

This repository provides a model integration between the ResNet classification model and the [Dataloop](https://dataloop.ai/) platform.

ResNet (Residual Networks) are a series of deep learning models designed for image classification tasks. They are known for their ability to train very deep networks by using residual connections, which help mitigate the vanishing gradient problem. This integration aims to provide a seamless workflow for utilizing ResNet models within the Dataloop platform.

## Requirements

* An account in the [Dataloop platform](https://console.dataloop.ai/)
* dtlpy
* torch>=1.8
* torchvision>0.9
* imgaug
* numpy
* tqdm

## Installation

To install the package and create the ResNet model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory) containing its training and validation subsets.

## Cloning

For instructions on how to clone the pretrained model for prediction, click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting).

## Training and Fine-tuning

For fine-tuning on a custom dataset, click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

### Editing the Configuration

To edit configurations via the platform, go to the ResNet page in the Model Management and edit the JSON file displayed there or, via the SDK, by editing the model configuration. Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more information.

The basic configurations included are:

* `epochs`: Number of epochs to train the model (default: `10`)
* `batch_size`: Batch size to be used during the training (default: `16`)