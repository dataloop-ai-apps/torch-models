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
                self.model.load_state_dict(torch.load(weights_filename, map_location=self.device))
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
        num_epochs = self.configuration.get('num_epochs', 50)
        batch_size = self.configuration.get('batch_size', 4)
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
        lr = self.configuration.get('learning_rate', 0.005)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        # Decay LR by a factor of 0.1 - slower decay for better convergence
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

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

        input_size = self.configuration.get('input_size', 520)
        debug_mode = self.configuration.get('debug_inference', False)

        preprocess = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                     torchvision.transforms.Resize((input_size, input_size),
                                                                                   interpolation=InterpolationMode.BILINEAR),
                                                     self.gray_to_rgb,
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])

        batch_annotations = list()

        for idx, img in enumerate(batch):
            orig_h, orig_w = img.shape[:2]  # Store original dimensions
            img_tensor = preprocess(img.astype('uint8')).unsqueeze(0).to(self.device)  # Add batch dimension
            collection = self.inference(img_tensor, orig_w, orig_h, input_size, 
                                        original_image=img, debug_mode=debug_mode, debug_idx=idx)
            batch_annotations.append(collection)

        return batch_annotations

    def inference(self, img_tensor, orig_w, orig_h, input_size, original_image=None, debug_mode=False, debug_idx=0):
        """
        Run inference on a single image tensor and return polygon annotations.
        
        The core prediction is simple (same as standalone):
            output = model(img)['out'][0]
            pred_mask = output.argmax(dim=0)  # class with highest score per pixel
        
        The extra steps convert the mask into Dataloop polygon annotations.
        """
        labels = list(self.model_entity.id_to_label_map.values())
        threshold = self.configuration.get('conf_threshold', 0.5)

        # === CORE PREDICTION ===
        with torch.no_grad():
            output = self.model(img_tensor)['out'][0]
        pred_mask = output.argmax(dim=0).cpu().numpy().astype(np.uint8)
        
        # === CONVERT MASK TO POLYGON ANNOTATIONS ===
        # Scale factors to map coordinates back to original image size
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size

        collection = dl.AnnotationCollection()
        
        # Process each class (skip background = 0)
        for class_idx in range(1, len(labels)):
            class_mask = (pred_mask == class_idx).astype(np.uint8)
            
            if class_mask.sum() == 0:  # No pixels for this class
                continue
            
            class_label = labels[class_idx]
            contours = self.extract_contours(class_mask)

            for polygon in contours:
                # Scale polygon coordinates back to original image size
                scaled_polygon = polygon.astype(np.float32)
                scaled_polygon[:, 0] *= scale_x
                scaled_polygon[:, 1] *= scale_y
                
                # Simple confidence: ratio of polygon area that matches predicted class
                poly_area = cv2.contourArea(polygon.astype(np.int32))
                confidence = min(1.0, poly_area / 100)  # Simple heuristic
                
                collection.add(
                    annotation_definition=dl.Polygon(geo=scaled_polygon, label=class_label),
                    model_info={'name': self.model_entity.name,
                                'confidence': confidence,
                                'model_id': self.model_entity.id}
                )
        
        # === DEBUG: Save visualization if enabled ===
        if debug_mode and original_image is not None:
            self._save_debug_image(original_image, pred_mask, collection, labels, debug_idx)
        
        return collection
    
    def _save_debug_image(self, original_image, pred_mask, collection, labels, debug_idx):
        """Save debug visualization with contours overlaid on image."""
        import matplotlib.pyplot as plt
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Predicted mask (colored)
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        mask_colored = np.zeros((*pred_mask.shape, 3))
        for class_idx in range(len(labels)):
            mask_colored[pred_mask == class_idx] = colors[class_idx][:3]
        axes[1].imshow(mask_colored)
        axes[1].set_title("Predicted Mask")
        axes[1].axis('off')
        
        # Original with contours overlaid
        overlay = original_image.copy()
        for ann in collection:
            geo = np.array(ann.annotation_definition.geo, dtype=np.int32)
            cv2.polylines(overlay, [geo], isClosed=True, color=(0, 255, 0), thickness=2)
        axes[2].imshow(overlay)
        axes[2].set_title("Contours on Image")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save locally
        debug_path = f"debug_inference_{debug_idx}.png"
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved debug image: {debug_path}")
        
        # Upload to dataset debug folder
        try:
            dataset = self.model_entity.dataset
            remote_path = f"/debug/inference_{debug_idx}.png"
            dataset.items.upload(local_path=debug_path, remote_path="/debug/")
            logger.info(f"Uploaded debug image to dataset: {remote_path}")
        except Exception as e:
            logger.warning(f"Could not upload debug image: {e}")

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
            report.upload(dataset=self.model_entity.dataset, remote_path="/.dataloop/reports",
                          remote_name=f"confusion_model_{self.model_entity.id}.json")

        except Exception:
            logger.warning('Failed creating confusion report! Continue without...')

    @staticmethod
    def gray_to_rgb(x):
        return x.convert('RGB')

    @staticmethod
    def dl_collate(batch):
        """
        Custom collate function for DataLoader.
        
        Note: We create a single combined semantic segmentation mask per image rather than
        separate binary masks per annotation. This ensures all masks have a consistent shape
        [1, H, W] across the batch, which is required for torch.stack() in the training loop.
        Previously, masks had shape [num_annotations, H, W] which caused RuntimeError when
        batching images with different annotation counts.
        """
        # Stack images - b['image'] should be (C, H, W) from DatasetGeneratorTorch
        ims = torch.stack([b['image'].float() for b in batch])
        tgs = list()
        
        for b in batch:
            # Create a single combined semantic segmentation mask
            # where each pixel contains the class index (0 = background)
            # b['image'].shape is (C, H, W), so shape[1:] gives (H, W)
            target_h, target_w = b['image'].shape[1:]
            combined_mask = np.zeros(shape=(target_h, target_w), dtype=np.int64)
            
            # Calculate original image dimensions from segment coordinates
            # Segments are in original image space, need to scale to resized image
            if len(b['segment']) > 0:
                all_segs = np.concatenate([np.asarray(s) for s in b['segment']], axis=0)
                orig_w = max(int(np.ceil(all_segs[:, 0].max())) + 1, 1)
                orig_h = max(int(np.ceil(all_segs[:, 1].max())) + 1, 1)
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
            else:
                scale_x, scale_y = 1.0, 1.0
            
            for seg, class_id in zip(b['segment'], b['class']):
                mask = np.zeros(shape=(target_h, target_w), dtype=np.uint8)
                # Scale segment coordinates from original to resized image space
                seg_array = np.asarray(seg).astype(np.float32)
                seg_array[:, 0] *= scale_x  # scale x
                seg_array[:, 1] *= scale_y  # scale y
                seg_scaled = seg_array.astype('int')
                
                mask = cv2.drawContours(
                    image=mask,
                    contours=[seg_scaled],
                    contourIdx=-1,
                    color=1,
                    thickness=-1
                )
                # Assign the class_id to pixels belonging to this segment
                combined_mask[mask == 1] = class_id
                
            boxes = torch.as_tensor(b['box'], dtype=torch.float32)
            # Store the combined mask with shape [1, H, W] for consistency
            masks = torch.as_tensor(combined_mask, dtype=torch.int64).unsqueeze(0)
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
    def extract_contours(mask, epsilon_factor=0.01):
        """
        Extract and simplify contours from a binary mask.
        
        Args:
            mask: Binary mask (H, W)
            epsilon_factor: Controls polygon simplification (higher = simpler polygon)
        """
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = []
        for contour in contours:
            # Approximate the contour to reduce jagged edges
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:  # Need at least 3 points for a valid polygon
                simplified_contours.append(approx.squeeze(axis=1).reshape(-1, 2))
        return simplified_contours
