import os
import json


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def find_readmes(base_path):
    readmes = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower() == "readme.md" and root != base_path:
                if "env" not in root.lower():
                    readmes[root] = read_file(os.path.join(root, file))
    return readmes


def write_main_readme(main_readme_path, readmes, manifest_data):
    with open(main_readme_path, "w", encoding="utf-8") as file:
        file.write(
            "# Torch Dataloop Model Adapters 🚀\n\n"
            "Welcome to the Torch Dataloop Model Adapters repository! 🎉 This collection features a variety of pre-trained models built using PyTorch, "
            "a leading open-source machine learning library. These models are designed to tackle a range of tasks including image classification, "
            "object detection, and semantic segmentation. This README provides a detailed overview of the available model adapters within the Dataloop platform. 📚\n\n"
            "## Documentation 📖\n\n"
            "For comprehensive documentation on model management, visit the [Dataloop Model Management Documentation](https://dataloop.ai/docs).\n"
            "For developer tutorials and guides, check out the [Developers Documentation](https://developers.dataloop.ai/tutorials/model_management/).\n\n"
            "## Marketplace 🛒\n\n"
            "Explore a wide range of Torch models available in the Dataloop [Marketplace](https://console.dataloop.ai/marketplace). "
            "The marketplace offers a variety of pre-trained models ready to be integrated into your projects, enabling you to leverage state-of-the-art machine learning capabilities with ease.\n\n"
            "## SDK Usage 🛠️\n\n"
            "### Cloning a Model to Your Project\n\n"
            "You can easily clone a pre-trained model to your project to start using it immediately. Here's how to get all the public models:\n\n"
            "```python\n"
            "import dtlpy as dl\n\n"
            "filters = dl.Filters(resource=dl.FILTERS_RESOURCE_MODEL)\n"
            "filters.add(field='scope', values='public')\n\n"
            "dl.models.list(filters=filters).print()\n"
            "```\n\n"
            "Once you've identified the model you want to use, clone it into your project:\n\n"
            "```python\n"
            "import dtlpy as dl\n\n"
            "public_model = dl.models.get(model_id='646dae2b6cd40e80856fe0f1')\n"
            "project = dl.projects.get('My Project')\n"
            "model = project.models.clone(from_model=public_model,\n"
            "                             model_name='my_pretrained_resnet_50',\n"
            "                             project_id=project.id)\n"
            "```\n\n"
            "### Fine-tuning a Model\n\n"
            "To fine-tune a model, connect it to your dataset and initiate training:\n\n"
            "```python\n"
            "dataset = project.datasets.get('Capybaras')\n"
            "train_filter = dl.Filters(field='dir', values='/train')\n"
            "validation_filter = dl.Filters(field='dir', values='/validation')\n"
            "custom_model = project.models.clone(from_model=public_model,\n"
            "                                    model_name='finetuning_mode',\n"
            "                                    dataset=dataset,\n"
            "                                    project_id=project.id,\n"
            "                                    train_filter=train_filter,\n"
            "                                    validation_filter=validation_filter)\n"
            "```\n\n"
            "Now you have a new model connected to your dataset, ready for training. For more information, visit the [Training Guide](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train).\n\n"
            "## PyTorch Model Adapters\n\n"
        )

        for root, content in readmes.items():
            # Determine the path relative to "adapters"
            relative_path = os.path.relpath(root, "adapters")
            path_parts = relative_path.split(os.sep)

            # Construct the summary based on the path depth
            if len(path_parts) == 1:
                summary = path_parts[0].title()
            else:
                summary = " ".join(part.title() for part in path_parts)

            file.write("<details>\n")
            file.write(f"<summary>{summary} Documentation</summary>\n\n")
            file.write(content)
            file.write("\n</details>\n\n")

        # Add Dataloop Manifest (DPK) Explanation
        file.write(
            "## Dataloop Manifest (DPK) Explanation 📜\n\n"
            "This section provides an explanation of the [DeepLabv3 manifest](adapters\\deeplabv3\\dataloop.json), which can be used as an example for a *model* application.\n\n"
            "### Dataloop Applications\n"
            "Dataloop Applications are extensions that integrate seamlessly into the Dataloop ecosystem, providing custom panels, SDK features, and components to enhance your workflow. "
            "For more information, visit the [Dataloop Applications Introduction](https://developers.dataloop.ai/tutorials/applications/introduction/chapter).\n\n"
            "### DPK (Dataloop Package Kit)\n"
            "The DPK is a comprehensive package that includes everything needed for your application to function within the Dataloop platform. "
            "It contains modules, panels, source code, tests, and the `dataloop.json` manifest, which acts as the application's blueprint.\n\n"
            "The Dataloop Manifest (DPK) provides metadata and configuration details for deploying and managing models on the Dataloop platform. "
            "Here's an explanation of the key components in the manifest:\n\n"
            "- **Name**: The identifier for the model package.\n"
            "- **Display Name**: A user-friendly name for the model.\n"
            "- **Version**: The version of the model package.\n"
            "- **Scope**: Defines the visibility of the model (e.g., public or private).\n"
            "- **Description**: A brief description of the model and its purpose.\n"
            "- **Provider**: The entity or framework providing the model.\n"
            "- **Deployed By**: The organization or platform deploying the model.\n"
            "- **License**: The licensing terms under which the model is distributed.\n"
            "- **Category**: The category or type of model (e.g., Model, Dataset).\n"
            "- **Computer Vision Task**: The specific computer vision task the model addresses (e.g., Semantic Segmentation).\n"
            "- **Media Type**: The type of media the model is designed to process (e.g., Image).\n\n"
            "### Codebase\n"
            "- **Type**: The type of code repository (e.g., git).\n"
            "- **Git Tag**: The specific tag or commit in the repository that corresponds to this version of the model.\n\n"
            "All codebase information can be removed if you are using local code.\n"
            "- **Git URL**: The URL of the git repository containing the model's code.\n"
            "### Components\n"
            "#### Compute Configurations\n"
            "Defines the computational resources and settings required to run the model, including pod type, concurrency, and autoscaling settings. "
            "Here is an example of one configuration, but more than one can be defined:\n\n"
        )

        # Example of one compute configuration
        compute_config = manifest_data["components"]["computeConfigs"][0]
        file.write(
            f"- **Name**: {compute_config['name']}\n"
            f"  - **Pod Type**: The type of pod used for deployment (e.g., regular-xs, gpu-t4).\n"
            f"  - **Concurrency**: The number of concurrent executions allowed.\n"
            f"  - **Runner Image**: The Docker image used to run the model.\n"
            f"  - **Autoscaler Type**: The type of autoscaler used (e.g., rabbitmq).\n"
            f"  - **Min Replicas**: The minimum number of pod replicas.\n"
            f"  - **Max Replicas**: The maximum number of pod replicas.\n"
            f"  - **Queue Length**: The length of the queue for processing tasks.\n\n"
        )

        file.write("#### Modules\n")
        for module in manifest_data["components"]["modules"]:
            file.write(
                f"- **Name**: {module['name']}\n"
                f"  - **Entry Point**: The main script or module to execute.\n"
                f"  - **Class Name**: The class within the entry point that implements the model logic.\n"
                f"  - **Compute Config**: The compute configuration associated with this module.\n"
                f"  - **Description**: A description of the module's functionality.\n\n"
            )

        file.write("#### Models\n")
        # Example of one model
        model = manifest_data["components"]["models"][0]
        file.write(
            f"- **Name**: {model['name']}\n"
            f"  - **Module Name**: The module that this model is part of.\n"
            f"  - **Scope**: The scope of the model (e.g., project-specific).\n"
            f"  - **Status**: The current status of the model (e.g., pre-trained).\n"
            f"  - **Configuration**: The configuration settings for the model, such as batch size and number of epochs.\n"
            f"  - **Input Type**: The type of input data the model accepts.\n"
            f"  - **Output Type**: The type of output data the model produces.\n"
            f"  - **Description**: A detailed description of the model's capabilities and use cases.\n"
            f"  - **Labels**: The set of labels or classes the model can predict.\n\n"
            "More than one model can be defined in the manifest.\n\n"
        )

        file.write(
            "## Contributions 🤝\n\n"
            "Help us improve! We welcome any contributions and suggestions to this repository.\n"
            "Feel free to open an issue for bug reports or feature requests.\n"
        )


def main():
    base_path = "."
    main_readme_path = "README.md"
    readmes = find_readmes(base_path)

    # Load the manifest data
    manifest_path = "adapters/deeplabv3/dataloop.json"
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest_data = json.load(manifest_file)

    write_main_readme(main_readme_path, readmes, manifest_data)


if __name__ == "__main__":
    main()
