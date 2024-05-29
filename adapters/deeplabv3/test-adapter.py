from model_adapter import ModelAdapter
import dtlpy as dl

if __name__ == '__main__':
    dl.setenv("rc")
    project = dl.projects.get(project_name="segmentation-models")
    datasets = project.datasets.get(dataset_name="ImageNet")
    item1 = datasets.items.get(item_id="664cb0bf360fa1a7da7d5d27")
    item2 = datasets.items.get(item_id="664cb0bf360fa130817d5d20")
    item3 = datasets.items.get(item_id="664cb0bee24e77757c67763c")
    persons = datasets.items.get(item_id="664f53b4b43b02764663655d")
    pytorch_exmp = datasets.items.get(item_id="6652d6a0f8fc22479583749a")

    # Load and predict
    model_entity = project.models.get(model_name="yolox-s")
    adapter = ModelAdapter(model_entity)
    adapter.predict_items([pytorch_exmp, item1, item3, persons])

    # Train and save
    project = dl.projects.get(project_name="segmentation-models")
    dataset1 = project.datasets.get(dataset_name="dog-cats")
    model_to_train = project.models.get(model_id="6655d7accf683639ae5e91f6")
    # import json
    #
    # subsets = {'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
    #            'validation': json.dumps(dl.Filters(field='dir', values='/val').prepare())}
    # dataset1.metadata['system']['subsets'] = subsets
    # dataset1.update(True)
    # if 'system' not in model_to_train.metadata:
    #     model_to_train.metadata['system'] = {}
    # # Set the subsets for training and validation
    # model_to_train.metadata['system']['subsets'] = {
    #     'train': dl.Filters(field='dir', values='/train').prepare(),
    #     'validation': dl.Filters(field='dir', values='/val').prepare()
    # }
    # model_to_train.update(True)

    model_to_train.labels = ['dog', 'cat']
    model_to_train.update(True)
    #
    # # resnet pretrained
    adapter1 = ModelAdapter(model_to_train)
    adapter1.train_model(model_to_train)

    # predicted with new weight:
    project = dl.projects.get(project_name="segmentation-models")
    datasets = project.datasets.get(dataset_name="dog-cats")
    item = datasets.items.get(item_id="6656f113eb4237fb401e2799")
    item1 = datasets.items.get(item_id="6656f107f90fbdd8a422c4d3")
    item2 = datasets.items.get(item_id="665592ef95c92d28416adbf9")
    item3 = datasets.items.get(item_id="665592ef58f16d5ecf574035")

    adapter1 = ModelAdapter(model_to_train)
    adapter1.predict_items([item, item1, item2, item3])
