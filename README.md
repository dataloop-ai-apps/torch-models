# Torch Dataloop Model Adapters

These are pytorch model adapters examples.

1. ResNet50 (resnet_adapter.py)

Full Model Management documentation [here](https://dataloop.ai/docs).
Developers Documentation is [here](https://developers.dataloop.ai/tutorials/model_management/).  

## Clone Model to a Project
You can clone the pretrained model to your project to work as-is.
First get all the public model:

```python
import dtlpy as dl
filters = dl.Filters(resource=dl.FILTERS_RESOURCE_MODEL)
filters.add(field='scope', values='public')

dl.models.list(filters=filters).print()
```

Select the pretrained model you want to clone and... clone:
```python
import dtlpy as dl
public_model = dl.models.get(model_id='646dae2b6cd40e80856fe0f1')
project = dl.projects.get('My Project')
model = project.models.clone(from_model=public_model,
                             model_name='my_pretrained_resnet_50',
                             project_id=project.id)
```

## Finetune
If you want to finetune the model, you'll need to connect your dataset and train:
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

Now you have a new model connected to your capybara dataset, and you can initiate a training execution.
More information [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train)

## Contributions

Help us get better! We welcome any contribution and suggestion to this repo.   
Open an issue for bug/features requests.
 
