from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch
from dtlpy.utilities.reports import Report, FigOptions, ConfusionMatrix
from torchvision.transforms.functional import InterpolationMode
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional
import seaborn as sns
import numpy as np
import dtlpy as dl
import torch.nn
import logging
import torch
import time
import tqdm
import cv2
import os

logger = logging.getLogger('segmentation-models-adapter')


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model
            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights = self.configuration.get('weights_filename', None)
        model_name = self.configuration.get('model_name', 'deeplabv3_resnet50')
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        if weights is not None:
            weights_filename = os.path.join(local_path, weights)
            if os.path.isfile(weights_filename):
                logger.info("Loading a model from {}".format(weights_filename))
                num_classes = len(self.model_entity.id_to_label_map.items())
                self.model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
                self.model.load_state_dict(torch.load(weights_filename, map_location=self.device, weights_only=False))
                logger.info("Loaded custom weights {}".format(weights_filename))
            else:
                raise Exception(
                    "No weights file found! Make sure you upload the weights file as an artifact to the model!")

        self.model.to(self.device)

        # Check if the first ID is not 'background'
        if self.model_entity.id_to_label_map.get(0) != 'background':
            logger.info("Changing id 0 to background")
            # Create a new id_to_label_map with 0 as 'background'
            new_id_to_label_map = {0: 'background'}

            # Adjust all other IDs by incrementing them by 1
            for idx, label in enumerate(self.model_entity.id_to_label_map.values(), start=1):
                new_id_to_label_map[idx] = label

            # Update the model entity's id_to_label_map
            self.model_entity.id_to_label_map = new_id_to_label_map
            self.model_entity.label_to_id_map = new_id_to_label_map
            self.model_entity.update()
            logger.info("Updated id_to_label_map and label_to_map_id with background as index 0")

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        torch.save(self.model.state_dict(), os.path.join(local_path, 'best.pth'))
        self.configuration['weights_filename'] = 'best.pth'

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the model to dump_path

        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        num_epochs = self.configuration.get('num_epochs', 100)
        batch_size = self.configuration.get('batch_size', 64)
        input_size = self.configuration.get('input_size', 520)
        augmentation = self.configuration.get('aug', True)
        on_epoch_end_callback = kwargs.get('on_epoch_end_callback')
        train_filter = self.model_entity.metadata['system']['subsets']['train']['filter']
        val_filter = self.model_entity.metadata['system']['subsets']['validation']['filter']

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'weights'), exist_ok=True)
        logger.info("Model set to train mode.")

        train_transformer = [torchvision.transforms.ToPILImage(),
                             self.gray_to_rgb,
                             torchvision.transforms.Resize((input_size, input_size),
                                                           interpolation=InterpolationMode.BILINEAR)]
        if augmentation:
            train_transformer += [
                torchvision.transforms.RandomHorizontalFlip(0.5),  # horizontal flip with a probability of 50%.
                torchvision.transforms.RandomVerticalFlip(0.2),  # vertical flip with a probability of 20%.
            ]

        train_transformer += [torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])]
        # DATA TRANSFORMERS
        data_transforms = {

            'train': train_transformer,
            'val': [
                torchvision.transforms.ToPILImage(),
                self.gray_to_rgb,
                torchvision.transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]

        }

        ####################
        # Prepare the data #
        ####################

        train_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'train'),
                                              filters=dl.Filters(custom_filter=train_filter),
                                              dataset_entity=self.model_entity.dataset,
                                              id_to_label_map=self.model_entity.id_to_label_map,
                                              label_to_id_map=self.model_entity.label_to_id_map,
                                              overwrite=False,
                                              to_mask=False,
                                              annotation_type=dl.AnnotationType.POLYGON,
                                              transforms=data_transforms['train']
                                              )

        val_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'validation'),
                                            filters=dl.Filters(custom_filter=val_filter),
                                            dataset_entity=self.model_entity.dataset,
                                            id_to_label_map=self.model_entity.id_to_label_map,
                                            label_to_id_map=self.model_entity.label_to_id_map,
                                            overwrite=False,
                                            to_mask=False,
                                            annotation_type=dl.AnnotationType.POLYGON,
                                            transforms=data_transforms['val']
                                            )

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=self.dl_collate),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=self.dl_collate,
                                         )}
        logger.debug("Train and Val data loaders created")

        #####################
        # Prepare the model #
        #####################

        n_classes = len(train_dataset.id_to_label_map)
        logger.info('Setting last layer for {} classes'.format(n_classes))
        self.model.classifier[-1] = torch.nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))

        criterion = torch.nn.CrossEntropyLoss()  # For multi-class segmentation
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        since = time.time()

        # early stopping params
        best_loss = np.inf
        best_acc = 0.0
        not_improving_epochs = 0
        patience_epochs = self.configuration.get("patience_epochs", 10)
        early_stop = self.configuration.get("early_stopping", False)
        end_training = False
        self.metrics = {'history': list()}
        self.model.to(device=self.device)

        ############
        # Training #
        ############

        logger.debug("Starting training")
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

                # Iterate over data
                with tqdm.tqdm(dataloaders[phase], unit="batch") as tepoch:
                    for batch in tepoch:
                        inputs = torch.stack(tuple(batch[0]), 0).to(self.device)
                        masks = torch.stack([item['masks'] for item in batch[1]]).squeeze(1).to(self.device)
                        masks = masks.long()
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)['out']
                            _, preds = torch.max(outputs, 1)

                            loss = criterion(outputs, masks)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        total += inputs.size(0)
                        running_loss += (loss.item() * inputs.size(0))
                        running_corrects += torch.sum(preds == masks).double().cpu().numpy()
                        epoch_acc = running_corrects / total
                        epoch_loss = running_loss / total
                        tepoch.set_postfix(loss=epoch_loss, accuracy=100. * epoch_acc)

                if phase == 'train':
                    exp_lr_scheduler.step()

                logger.info(
                    f'Epoch {epoch}/{num_epochs} - {phase} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, '
                    f'Duration {(time.time() - epoch_time):.2f}')
                plots = [dl.PlotSample(figure='loss',
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

                        logger.info(
                            f'Validation loss decreased ({best_loss:.6f} --> {epoch_loss:.6f}).  Saving model ...')
                        torch.save(self.model.state_dict(), os.path.join(output_path, 'best.pth'))  # saving ckpt

                    else:
                        not_improving_epochs += 1
                    if not_improving_epochs > patience_epochs and early_stop:
                        end_training = True
                        logger.info(f'EarlyStopping counter: {not_improving_epochs} out of {patience_epochs}')

            #############
            # Callbacks #
            #############
            if on_epoch_end_callback is not None:
                on_epoch_end_callback(i_epoch=epoch,
                                      n_epoch=num_epochs)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best val loss: {:4f}, best acc: {:4f}'.format(best_loss, best_acc))
        labels = list(train_dataset.id_to_label_map.values())
        self.cm_report(dataloaders=dataloaders, labels=labels)

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        self.model.eval()

        # input_size = self.configuration.get('input_size', 224)

        preprocess = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                     # torchvision.transforms.Resize((input_size, input_size)),
                                                     self.gray_to_rgb,
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])

        batch_annotations = list()

        for img in batch:
            img_tensor = preprocess(img.astype('uint8')).unsqueeze(0).to(self.device)  # Add batch dimension
            collection = self.inference(img_tensor)
            batch_annotations.append(collection)

        return batch_annotations

    def inference(self, img_tensor):
        labels = list(self.model_entity.id_to_label_map.values())
        threshold = self.configuration.get('conf_threshold', 0.8)

        with torch.no_grad():  # Forward pass through the model
            img_output = self.model(img_tensor)['out'][0]
        probs = torch.softmax(img_output, dim=0)
        output_predictions = probs.argmax(dim=0)

        # Get the unique class indices in the predictions excluding class index 0
        unique_class_indices = torch.unique(output_predictions.flatten())
        unique_class_indices = unique_class_indices[unique_class_indices != 0]

        collection = dl.AnnotationCollection()
        for class_idx in unique_class_indices:
            confidence = probs[class_idx].cpu().numpy().max()
            if confidence < threshold:  # Skip if confidence is below the threshold
                logger.info(f"Confidence:{confidence} , is lower than threshold: {threshold}")
                continue

            class_label = labels[class_idx]
            class_mask = (output_predictions == class_idx).cpu().numpy().astype(np.uint8)
            contours = self.extract_contours(class_mask)
            # self.plot_contours(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), contours[0])

            for polygon in contours:
                collection.add(annotation_definition=dl.Polygon(geo=polygon, label=class_label),
                               model_info={'name': self.model_entity.name,
                                           'confidence': confidence,
                                           'model_id': self.model_entity.id})
        return collection

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

    def cm_report(self, dataloaders, labels):
        ####################
        # Confusion Matrix #
        ####################
        try:
            y_true_total = list()
            y_pred_total = list()
            colors = sns.color_palette("rocket", as_cmap=True)
            report = Report(ncols=1, nrows=3)

            for phase in ['train', 'val']:
                y_true = list()
                y_pred = list()

                for batch in dataloaders[phase]:
                    inputs = torch.stack(tuple(batch[0]), 0).to(self.device)
                    masks = torch.stack([item['masks'] for item in batch[1]]).squeeze(1).to(self.device).long()

                    with torch.set_grad_enabled(False):
                        outputs = self.model(inputs)['out']
                        _, preds = torch.max(outputs, 1)

                    y_true.extend(masks.cpu().numpy().flatten())
                    y_pred.extend(preds.cpu().numpy().flatten())

                    # Add to the overall true and predicted labels
                    y_true_total.extend(y_true)
                    y_pred_total.extend(y_pred)

                data = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize='true')
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
                    confusion.title = "Val Confusion"
                    report.add(fig=confusion, icol=0, irow=1)

            # Overall confusion matrix
            data = confusion_matrix(y_true_total, y_pred_total, labels=range(len(labels)), normalize='true')
            color_map = colors(data)
            href_map = [[self.model_entity.dataset.platform_url for _ in range(data.shape[0])] for _ in
                        range(data.shape[0])]
            confusion = ConfusionMatrix(
                title="Overall Confusion",
                labels=labels,
                data=data,
                color_map=color_map,
                href_map=href_map,
                options=FigOptions(rows_per_page=100, x_title="True", y_title="Pred")
            )
            report.add(fig=confusion, icol=0, irow=2)

            # Upload the report to a dataset
            report.upload(dataset=self.model_entity.dataset, remote_path="/reports",
                          remote_name=f"confusion_model_{self.model_entity.id}.json")

        except Exception:
            logger.warning('Failed creating confusion report! Continue without...')

    @staticmethod
    def gray_to_rgb(x):
        return x.convert('RGB')

    @staticmethod
    def dl_collate(batch):
        ims = torch.stack([torch.transpose(b['image'].float(), 2, 1) for b in batch])
        tgs = list()
        for b in batch:
            masks = list()
            for seg in b['segment']:
                mask = np.zeros(shape=b['image'].shape[1:])
                mask = cv2.drawContours(
                    image=mask,
                    contours=[np.asarray(seg).astype('int')],
                    contourIdx=-1,
                    color=True,
                    thickness=-1
                )
                masks.append(mask)
            boxes = torch.as_tensor(b['box'], dtype=torch.float32)
            masks = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)
            labels = torch.Tensor(b['class']).to(torch.int64)
            tgs.append(
                {'boxes': boxes,
                 'area': [(box[3] - box[1]) * (box[2] - box[0]) for box in boxes],
                 "iscrowd": [True] * len(labels),
                 'labels': labels,
                 'masks': masks,
                 'image_id': b['item_id']}
            )

        return ims, tgs

    @staticmethod
    def plot_contours(image, contours):
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')

        plt.plot(contours[:, 0], contours[:, 1], linewidth=2, label='Contour')
        plt.title('Contours on Image')
        plt.axis('off')
        plt.legend()
        plt.show()

    @staticmethod
    def extract_contours(mask):
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = [contour.squeeze(axis=1).reshape(-1, 2) for contour in contours]
        return contours
