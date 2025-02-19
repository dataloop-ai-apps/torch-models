# Torch Dataloop Model Adapters 🚀

Welcome to the Torch Dataloop Model Adapters repository! 🎉 This collection features a variety of pre-trained models built using PyTorch, a leading open-source machine learning library. These models are designed to tackle a range of tasks including image classification, object detection, and semantic segmentation. This README provides a detailed overview of the available model adapters within the Dataloop platform. 📚

## Documentation 📖

For comprehensive documentation on model management, visit the [Dataloop Model Management Documentation](https://dataloop.ai/docs).
For developer tutorials and guides, check out the [Developers Documentation](https://developers.dataloop.ai/tutorials/model_management/).

## Marketplace 🛒

Explore a wide range of Torch models available in the Dataloop [Marketplace](https://console.dataloop.ai/marketplace). The marketplace offers a variety of pre-trained models ready to be integrated into your projects, enabling you to leverage state-of-the-art machine learning capabilities with ease.

## SDK Usage 🛠️

### Cloning a Model to Your Project

You can easily clone a pre-trained model to your project to start using it immediately. Here's how to get all the public models:

```python
import dtlpy as dl

filters = dl.Filters(resource=dl.FILTERS_RESOURCE_MODEL)
filters.add(field='scope', values='public')

dl.models.list(filters=filters).print()
```

Once you've identified the model you want to use, clone it into your project:

```python
import dtlpy as dl

public_model = dl.models.get(model_id='646dae2b6cd40e80856fe0f1')
project = dl.projects.get('My Project')
model = project.models.clone(from_model=public_model,
                             model_name='my_pretrained_resnet_50',
                             project_id=project.id)
```

### Fine-tuning a Model

To fine-tune a model, connect it to your dataset and initiate training:

```python
dataset = project.datasets.get('Capybaras')
train_filter = dl.Filters(field='dir', values='/train')
validation_filter = dl.Filters(field='dir', values='/validation')
custom_model = project.models.clone(from_model=public_model,
                                    model_name='finetuning_mode',
                                    dataset=dataset,
                                    project_id=project.id,
                                    train_filter=train_filter,
                                    validation_filter=validation_filter)
```

Now you have a new model connected to your dataset, ready for training. For more information, visit the [Training Guide](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train).

## PyTorch Model Adapters

<details>
<summary>Deeplabv3 Documentation</summary>

# DeepLabV3 Semantic Segmentation Model Adapter

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

To install the package and create the DeepLabV3 model adapter, you will need
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

To edit configurations via the platform, go to the DeepLabV3 page in the Model Management and edit the json
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



</details>

<details>
<summary>Resnet Documentation</summary>

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
</details>

<details>
<summary>Speechbrain Classification Documentation</summary>

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
</details>

<details>
<summary>Timm Classification Documentation</summary>

# TIMM Model Adapter

## Introduction

This repo is a model integration between TIMM models for classification [TIMM Models](https://huggingface.co/timm) and [Dataloop](https://dataloop.ai/).

PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

## Models available

Various models and their variants can be used. To switch models, edit the model name in the configuration after installation from the Marketplace. The default models are:  trained or finetuned onImageNet1k (1k at the end of the model).

- EfficientNet (default: 'efficientnetv2_rw_s.ra2_in1k')
- HRNet (default: 'hrnet_w18.ms_aug_in1k')
- MobileNet (default: 'mobilenetv3_large_100.ra_in1k')

Other variants from these models can be found [here](https://huggingface.co/models?pipeline_tag=image-classification&library=timm) and used if trained or fine-tuned on ImageNet1k.

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
</details>

## Dataloop Manifest (DPK) Explanation 📜

This section provides an explanation of the [DeepLabv3 manifest](adapters\deeplabv3\dataloop.json), which can be used as an example for a *model* application.

### Dataloop Applications
Dataloop Applications are extensions that integrate seamlessly into the Dataloop ecosystem, providing custom panels, SDK features, and components to enhance your workflow. For more information, visit the [Dataloop Applications Introduction](https://developers.dataloop.ai/tutorials/applications/introduction/chapter).

### DPK (Dataloop Package Kit)
The DPK is a comprehensive package that includes everything needed for your application to function within the Dataloop platform. It contains modules, panels, source code, tests, and the `dataloop.json` manifest, which acts as the application's blueprint.

The Dataloop Manifest (DPK) provides metadata and configuration details for deploying and managing models on the Dataloop platform. Here's an explanation of the key components in the manifest:

- **Name**: The identifier for the model package.
- **Display Name**: A user-friendly name for the model.
- **Version**: The version of the model package.
- **Scope**: Defines the visibility of the model (e.g., public or private).
- **Description**: A brief description of the model and its purpose.
- **Provider**: The entity or framework providing the model.
- **Deployed By**: The organization or platform deploying the model.
- **License**: The licensing terms under which the model is distributed.
- **Category**: The category or type of model (e.g., Model, Dataset).
- **Computer Vision Task**: The specific computer vision task the model addresses (e.g., Semantic Segmentation).
- **Media Type**: The type of media the model is designed to process (e.g., Image).

### Codebase
- **Type**: The type of code repository (e.g., git).
- **Git Tag**: The specific tag or commit in the repository that corresponds to this version of the model.

All codebase information can be removed if you are using local code.
- **Git URL**: The URL of the git repository containing the model's code.
### Components
#### Compute Configurations
Defines the computational resources and settings required to run the model, including pod type, concurrency, and autoscaling settings. Here is an example of one configuration, but more than one can be defined:

- **Name**: deeplab-deploy
  - **Pod Type**: The type of pod used for deployment (e.g., regular-xs, gpu-t4).
  - **Concurrency**: The number of concurrent executions allowed.
  - **Runner Image**: The Docker image used to run the model.
  - **Autoscaler Type**: The type of autoscaler used (e.g., rabbitmq).
  - **Min Replicas**: The minimum number of pod replicas.
  - **Max Replicas**: The maximum number of pod replicas.
  - **Queue Length**: The length of the queue for processing tasks.

#### Modules
- **Name**: deeplab-module
  - **Entry Point**: The main script or module to execute.
  - **Class Name**: The class within the entry point that implements the model logic.
  - **Compute Config**: The compute configuration associated with this module.
  - **Description**: A description of the module's functionality.

#### Models
- **Name**: pretrained-deeplab-resnet50
  - **Module Name**: The module that this model is part of.
  - **Scope**: The scope of the model (e.g., project-specific).
  - **Status**: The current status of the model (e.g., pre-trained).
  - **Configuration**: The configuration settings for the model, such as batch size and number of epochs.
  - **Input Type**: The type of input data the model accepts.
  - **Output Type**: The type of output data the model produces.
  - **Description**: A detailed description of the model's capabilities and use cases.
  - **Labels**: The set of labels or classes the model can predict.

More than one model can be defined in the manifest.

## Contributions 🤝

Help us improve! We welcome any contributions and suggestions to this repository.
Feel free to open an issue for bug reports or feature requests.
