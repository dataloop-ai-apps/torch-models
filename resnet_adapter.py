from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional
import pandas as pd
import numpy as np
import dtlpy as dl
import torch.nn
import logging
import torch
import time
import copy
import tqdm
import os

from dtlpy.utilities.dataset_generators.dataset_generator import collate_torch
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

logger = logging.getLogger('resnet-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for ResNet classification',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and model entities with model code implementation
    """

    def __init__(self, model_entity=None):
        if not isinstance(model_entity, dl.Model):
            # pending fix DAT-31398
            if isinstance(model_entity, str):
                model_entity = dl.models.get(model_id=model_entity)
            if isinstance(model_entity, dict) and 'model_id' in model_entity:
                model_entity = dl.models.get(model_id=model_entity['model_id'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ModelAdapter, self).__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = self.model_entity.configuration.get('weights_filename', 'model.pth')
        # load model arch and state
        model_path = os.path.join(local_path, weights_filename)
        logger.info("Loading a model from {}".format(local_path))
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        # How to load the label_map from loaded model
        logger.info("Loaded model from {} successfully".format(model_path))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', 'model.pth')
        torch.save(self.model, os.path.join(local_path, weights_filename))
        self.configuration['weights_filename'] = weights_filename

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the model to dump_path

            Virtual method - need to implement
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        num_epochs = self.configuration.get('num_epochs', 10)
        batch_size = self.configuration.get('batch_size', 64)
        input_size = self.configuration.get('input_size', 256)
        on_epoch_end_callback = kwargs.get('on_epoch_end_callback')

        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # DATA TRANSFORMERS
        def gray_to_rgb(x):
            return x.convert('RGB')

        data_transforms = {
            'train': [
                iaa.Resize({"height": input_size, "width": input_size}),
                iaa.flip.Fliplr(p=0.5),
                iaa.flip.Flipud(p=0.2),
                torchvision.transforms.ToPILImage(),
                gray_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ],
            'val': [torchvision.transforms.ToPILImage(),
                    gray_to_rgb,
                    torchvision.transforms.Resize((input_size, input_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

        }

        ####################
        # Prepare the data #
        ####################
        train_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'train'),
                                              dataset_entity=self.model_entity.dataset,
                                              annotation_type=dl.AnnotationType.CLASSIFICATION,
                                              transforms=data_transforms['train'],
                                              id_to_label_map=self.model_entity.id_to_label_map,
                                              class_balancing=True
                                              )
        val_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'validation'),
                                            dataset_entity=self.model_entity.dataset,
                                            annotation_type=dl.AnnotationType.CLASSIFICATION,
                                            transforms=data_transforms['val'],
                                            id_to_label_map=self.model_entity.id_to_label_map,
                                            )

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_torch),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         collate_fn=collate_torch,
                                         shuffle=True)}
        #################
        # prepare model #
        #################
        n_classes = len(train_dataset.id_to_label_map)
        logger.info('Setting last layer for {} classes'.format(n_classes))
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

        criterion = torch.nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # early stopping params
        best_loss = np.inf
        best_acc = 0.0
        not_improving_epochs = 0
        patience_epochs = 7
        end_training = False
        self.metrics = {'history': list()}
        #####
        self.model.to(device=self.device)
        for epoch in range(num_epochs):
            if end_training:
                break
            logger.info('Epoch {}/{} Start...'.format(epoch, num_epochs))
            epoch_time = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                total = 0

                # Iterate over data.
                with tqdm.tqdm(dataloaders[phase], unit="batch") as tepoch:
                    for batch in tepoch:
                        inputs = torch.stack(tuple(batch['image']), 0).to(self.device)
                        labels = torch.stack(tuple(batch['annotations']), 0).to(self.device).long().squeeze(1)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        total += inputs.size(0)
                        running_loss += (loss.item() * inputs.size(0))
                        running_corrects += torch.sum(preds == labels.data).double().cpu().numpy()
                        epoch_acc = running_corrects / total
                        epoch_loss = running_loss / total
                        tepoch.set_postfix(loss=epoch_loss, accuracy=100. * epoch_acc)

                if phase == 'train':
                    exp_lr_scheduler.step()

                logger.info(
                    f'Epoch {epoch}/{num_epochs} - {phase} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Duration {(time.time() - epoch_time):.2f}')
                # deep copy the model
                plots = [
                    dl.PlotSample(figure='loss',
                                  legend=phase,
                                  x=epoch,
                                  y=epoch_loss),
                    dl.PlotSample(figure='accuracy',
                                  legend=phase,
                                  x=epoch,
                                  y=epoch_acc)

                ]
                self.model_entity.metrics.create(samples=plots,
                                                 dataset_id=self.model_entity.dataset_id)
                self.metrics['history'].append({'phase': phase,
                                                'epoch': epoch,
                                                'loss': epoch_loss,
                                                'acc': epoch_acc})
                if phase == 'val':
                    if epoch_loss < best_loss:
                        not_improving_epochs = 0
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        logger.info(
                            f'Validation loss decreased ({best_loss:.6f} --> {epoch_loss:.6f}).  Saving model ...')
                        torch.save(self.model, os.path.join(output_path, 'best.pth'))
                        # self.model_entity.bucket.sync(local_path=output_path)

                    else:
                        not_improving_epochs += 1
                    if not_improving_epochs > patience_epochs:
                        end_training = True
                        logger.info(f'EarlyStopping counter: {not_improving_epochs} out of {patience_epochs}')
                ###############
                # save debugs #
                ###############

            #############
            # Callbacks #
            #############
            if on_epoch_end_callback is not None:
                on_epoch_end_callback(i_epoch=epoch,
                                      n_epoch=num_epochs)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best val loss: {:4f}, best acc: {:4f}'.format(best_loss, best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        #####################
        # Confusion Shebang #
        #####################
        try:
            from dtlpy.utilities.reports import Report, FigOptions, ConfusionMatrix
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            y_true_total = list()
            y_pred_total = list()
            labels = list(train_dataset.id_to_label_map.values())
            colors = sns.color_palette("rocket", as_cmap=True)
            report = Report(ncols=1, nrows=3)
            for phase in ['train', 'val']:
                y_true = list()
                y_pred = list()
                for batch in dataloaders[phase]:
                    xs = batch['image'].to(self.device)
                    ys = batch['annotations'].to(self.device)
                    y_true.extend([train_dataset.id_to_label_map[int(y)] for y in ys])
                    with torch.set_grad_enabled(False):
                        outputs = self.model(xs)
                        _, preds = torch.max(outputs, 1)
                    y_pred.extend([train_dataset.id_to_label_map[int(l)] for l in preds])
                y_true_total.extend(y_true)
                y_pred_total.extend(y_pred)
                data = confusion_matrix(y_true, y_pred,
                                        labels=labels,
                                        normalize='true')
                color_map = colors(data)
                href_map = [[self.model_entity.dataset.platform_url for _ in range(data.shape[0])] for _ in
                            range(data.shape[0])]
                confusion = ConfusionMatrix(title="Confusion",
                                            labels=labels,
                                            data=data,
                                            color_map=color_map,
                                            href_map=href_map,
                                            options=FigOptions(rows_per_page=100,
                                                               x_title="true",
                                                               y_title="pred"))
                if phase == 'train':
                    confusion.title = "Train Confusion"
                    report.add(fig=confusion, icol=0, irow=0)
                else:
                    confusion.title = "Train Confusion"
                    report.add(fig=confusion, icol=0, irow=1)
            data = confusion_matrix(y_true_total, y_pred_total,
                                    labels=labels,
                                    normalize='true')
            color_map = colors(data)
            href_map = [[self.model_entity.dataset.platform_url for _ in range(data.shape[0])] for _ in
                        range(data.shape[0])]
            confusion = ConfusionMatrix(title="Overall Confusion",
                                        labels=labels,
                                        data=data,
                                        color_map=color_map,
                                        href_map=href_map,
                                        options=FigOptions(rows_per_page=100,
                                                           x_title="true",
                                                           y_title="pred"))
            # Add the figures
            report.add(fig=confusion, icol=0, irow=2)
            # Upload the report to a dataset
            report.upload(dataset=self.model_entity.dataset,
                          remote_path="/reports",
                          remote_name=f"confusion_model_{self.model_entity.id}.json")
        except Exception:
            logger.warning('Failed creating shebang confusion report! Continue without...')

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        input_size = self.configuration.get('input_size', 256)
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(input_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
            ]
        )
        img_tensors = [preprocess(img.astype('uint8')) for img in batch]
        batch_tensor = torch.stack(img_tensors).to(self.device)
        batch_output = self.model(batch_tensor)
        batch_predictions = torch.nn.functional.softmax(batch_output, dim=1)
        batch_annotations = list()
        for img_prediction in batch_predictions:
            pred_score, high_pred_index = torch.max(img_prediction, 0)
            pred_label = self.model_entity.id_to_label_map.get(int(high_pred_index.item()), 'UNKNOWN')
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=pred_label),
                           model_info={'name': self.model_entity.name,
                                       'confidence': pred_score.item(),
                                       'model_id': self.model_entity.id,
                                       'dataset_id': self.model_entity.dataset_id})
            logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
            batch_annotations.append(collection)
        return batch_annotations

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        ...


def train():
    adapter = ModelAdapter()
    model = dl.models.get(model_id='63231d9982533076a48c685d')
    adapter.load_from_model(model)
    adapter.train_model(model)


def _get_imagenet_label_json():
    import json
    with open('imagenet_labels.json', 'r') as fh:
        labels = json.load(fh)
    return list(labels.values())


def package_creation():
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'model.pth',
                                                                 'input_size': 256},
                                          output_type=dl.AnnotationType.CLASSIFICATION,
                                          )
    module = dl.PackageModule.from_entry_point(entry_point='resnet_adapter.py')
    package = project.packages.push(package_name='resnet',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop ResNet implemented in pytorch',
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/torch-models',
                                                            git_tag='main'),
                                    modules=[module],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                        runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.7',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    return package


def model_creation(package: dl.Package, resnet_ver='50'):
    model = package.models.create(model_name='pretrained-resnet{}'.format(resnet_ver),
                                  description='resnset{} pretrained on imagenet'.format(resnet_ver),
                                  tags=['pretrained', 'imagenet'],
                                  dataset_id=None,
                                  scope='public',
                                  # scope='project',
                                  model_artifacts=[dl.LinkArtifact(
                                      url='https://storage.googleapis.com/model-mgmt-snapshots/ResNet50/model.pth',
                                      filename='model.pth')],
                                  status='trained',
                                  configuration={'weights_filename': 'model.pth',
                                                 'batch_size': 16,
                                                 'num_epochs': 10},
                                  project_id=project.id,
                                  labels=_get_imagenet_label_json(),
                                  )
    return model


if __name__ == "__main__":
    env = 'rc'
    project_name = 'DataloopModels'
    dl.setenv(env)
    project = dl.projects.get(project_name)
    # package = project.packages.get('resnet')
    # package.artifacts.list()
    # model_creation(package=package)
