from model_adapter import ModelAdapter
import dtlpy as dl

if __name__ == '__main__':
    # Predict with pre-trained weights
    project = dl.projects.get(project_name="")
    dataset = project.datasets.get(dataset_name="")
    model = project.models.get(model_id="")
    item = dataset.items.get(item_id="")
    adapter = ModelAdapter(model)
    adapter.predict_items([item])

    # Train
    project = dl.projects.get(project_name="")
    model = project.models.get(model_id="")
    adapter = ModelAdapter(model)
    adapter.train_model()

    # Predict with new weights
    project = dl.projects.get(project_name="")
    dataset = project.datasets.get(dataset_name="")
    model = project.models.get(model_id="")
    item = dataset.items.get(item_id="")
    adapter = ModelAdapter(model)
    adapter.predict_items([item])

