import os
import logging
import cv2
import torch
import torchvision
import numpy as np
import dtlpy as dl

from PIL import Image
from glob import glob
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch
from utils.engine import train_one_epoch, evaluate

logger = logging.getLogger('FasterRCNNAdapter')


@dl.Package.decorators.module(
    description='Model Adapter for FasterRCNN object detection',
    name='model-adapter',
    init_inputs={'model_entity': dl.Model}
    )
class FasterRCNNAdapter(dl.BaseModelAdapter):

    def get_model_instance_segmentation(self, num_classes):
        """
        This function initializes the pre-trained model with the correct parameters. It creates a FastRCNNPredictor
        whose hidden layers will have their size determined by the `hidden_layer` parameter in the model config. It then
        replaces the last layer by a new one with the size determined by num_classes, returning the MaskRCNN predictor
        with the expected sizes.

        :param num_classes: Number of classes that the model can predict
        :returns: faster-rcnn model with last layer replaced to reflect the number of classes parameter
        """
        hidden_layer = self.configuration.get("hidden_layer", 256)

        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
            )
        return model

    @staticmethod
    def dl_collate(batch):
        """
        The collate function here is used to pre-process batches before they are sent to the model.

        :param: batch: Input batches: The batches include:
            -image: a tensor with the input image
            -segment: the list of coordinates of each annotated object
            -box: the coordinates for the box containing each segmented object
            -class: the integer with the id of the class of each object
            -item_id: the reference to the item id
        :returns: img: tensor with the batch of images with channels in the order expected by the model
        :returns: tgs: a list of dictionaries, each dictionary describing one annotation in COCO format.
        """
        # The order of the channels expected by the model is different from what we receive, it needs transposing:
        ims = torch.stack([torch.transpose(b['image'].float(), 2, 1) for b in batch])
        tgs = list()
        for b in batch:
            masks = list()
            for seg in b['segment']:
                # The annotations we have are polygons, we need to generate a binary mask based on them:
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
                 "iscrowd": [True] * len(labels),  # Setting to true to use segmentation masks
                 'labels': labels,
                 'masks': masks,
                 'image_id': b['item_id']}
                )

        return ims, tgs

    def load(self, local_path, **kwargs):
        num_classes = len(self.model_entity.labels)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_filename = os.path.join(local_path, self.configuration.get('model_filename'))

        self.model = self.get_model_instance_segmentation(num_classes)
        os.makedirs(local_path, exist_ok=True)
        if model_filename is None:
            logger.info("No model weights file specified, loading default weights.")
        elif os.path.exists(model_filename):
            logger.info("Loading saved weights")
            self.model.load_state_dict(torch.load(model_filename, map_location=device))
        else:
            raise FileNotFoundError("Model weights file not found!")
        self.model.to(device)

    def save(self, local_path, **kwargs):
        model_filename = os.path.join(local_path, self.configuration.get('model_filename', 'weights/best.pt'))
        torch.save(self.model.state_dict(), model_filename)
        logger.info(f"Saved state dict at {model_filename}")
        self.configuration.update({'model_filename': 'weights/best.pt'})

    def prepare_item_func(self, item: dl.entities.Item):
        img_size = self.configuration.get("input_size", 256)
        buffer = item.download(save_locally=False)
        image = np.asarray(Image.open(buffer))
        if image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
            image = np.resize(image, (3, img_size, img_size))
        return image

    def predict(self, batch, **kwargs):
        # Reading configs:
        conf_threshold = self.configuration.get('conf_threshold', 0.5)
        id_to_label_map = self.configuration['id_to_label_map']

        self.model.eval()
        logger.info("Model set to evaluation mode")
        results = self.model(torch.Tensor(batch))
        logger.info("Batch prediction finished")
        batch_annotations = list()
        logger.info("Creating annotations based on predictions")
        for i_img, result in enumerate(results):
            logger.info(f"Annotations for item #{i_img}. Total number of detections: {len(result['labels'])}")
            image_annotations = dl.AnnotationCollection()
            for i_pred in range(len(result['labels'])):
                logger.info(f"Detection #{i_pred} for item #{i_img}")
                score = float(result['scores'][i_pred])
                if score < conf_threshold:
                    logger.info(
                        f"Ignoring detection, because its confidence ({score}) "
                        f"is lower than the threshold of {conf_threshold}"
                        )
                    continue
                cls = int(result['labels'][i_pred])
                mask = result['masks'][i_pred].cpu().detach().numpy().squeeze()
                logger.info(f"Class: {id_to_label_map[str(cls)]}, confidence: {score}")
                image_annotations.add(
                    annotation_definition=dl.Polygon.from_segmentation(
                        mask=mask,
                        label=id_to_label_map[str(cls)]
                        ),
                    model_info={'name': self.model_entity.name,
                                'model_id': self.model_entity.id,
                                'confidence': score}
                    )
            batch_annotations.append(image_annotations)
        logger.info("Annotations created successfully")
        return batch_annotations

    def train(self, data_path, output_path, **kwargs):
        # Reading configs:
        num_epochs = self.configuration.get("num_epochs", 10)
        input_size = self.configuration.get("input_size", 256)
        train_batch_size = self.configuration.get("train_batch_size", 12)
        val_batch_size = self.configuration.get("val_batch_size", 1)
        dataloader_num_workers = self.configuration.get("num_workers", 0)
        optim_learning_rate = self.configuration.get('learning_rate', 0.005)
        optim_momentum = self.configuration.get('momentum', 0.9)
        optim_weight_decay = self.configuration.get('weight_decay', 0.0005)
        scheduler_step_size = self.configuration.get('step_size', 3)
        scheduler_gamma = self.configuration.get('gamma', 0.1)
        id_to_label_map = self.model_entity.configuration.get("id_to_label_map")
        label_to_id_map = self.model_entity.configuration.get("label_to_id_map")
        train_filter = self.model_entity.metadata['system']['subsets']['train']['filter']
        val_filter = self.model_entity.metadata['system']['subsets']['validation']['filter']
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Using device: {device}")

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'weights'), exist_ok=True)
        logger.info("Model set to train mode.")

        def get_transform(train=True):
            transforms = list()
            transforms.append(T.ToTensor())
            if train:
                transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.Resize((input_size, input_size)))
            return T.Compose(transforms)

        logger.debug("Trainset generator created")
        train_dataset = DatasetGeneratorTorch(
            data_path=os.path.join(data_path, 'train'),
            filters=dl.Filters(custom_filter=train_filter),
            dataset_entity=self.model_entity.dataset,
            id_to_label_map=id_to_label_map,
            label_to_id_map=label_to_id_map,
            overwrite=False,
            to_mask=False,
            annotation_type=dl.AnnotationType.POLYGON,
            transforms=get_transform()
            )
        val_dataset = DatasetGeneratorTorch(
            data_path=os.path.join(data_path, 'validation'),
            filters=dl.Filters(custom_filter=val_filter),
            dataset_entity=self.model_entity.dataset,
            id_to_label_map=id_to_label_map,
            label_to_id_map=label_to_id_map,
            overwrite=False,
            to_mask=False,
            annotation_type=dl.AnnotationType.POLYGON,
            transforms=get_transform(False)
            )

        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            train_batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            collate_fn=self.dl_collate
            )
        logger.debug("Train data loader created")
        data_loader_test = torch.utils.data.DataLoader(
            val_dataset,
            val_batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            collate_fn=self.dl_collate
            )
        logger.debug("Val data loader created")
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=optim_learning_rate,
            momentum=optim_momentum,
            weight_decay=optim_weight_decay
            )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
            )
        logger.debug("Starting training")

        def epoch_end_callback(metrics, epoch):
            samples = list()
            for meter, value in metrics.meters.items():
                if 'loss' in meter:
                    legend = 'val' if 'val' in meter else 'train'
                    figure = meter.split('val_')[-1] if legend == 'val' else meter
                    samples.append(dl.PlotSample(figure=figure, legend=legend, x=epoch, y=value.value))
            self.model_entity.metrics.create(samples, dataset_id=self.model_entity.dataset_id)

        def eval_results_callback(metrics, epoch):
            samples = list()
            for iou_metric, coco_eval in metrics.coco_eval.items():
                stats = coco_eval.stats
                for i, stat in enumerate(stats):
                    metric_name = "AP" if i <= 5 else "AR"
                    if i == 1:
                        iou_thresh = "@IoU=0.5"
                    elif i == 2:
                        iou_thresh = "@IoU=0.75"
                    else:
                        iou_thresh = "@IoU=0.5:0.95"
                    if i in [3, 9]:
                        area = "|area=small"
                    elif i in [4, 10]:
                        area = "|area=medium"
                    elif i in [5, 11]:
                        area = "|area=large"
                    else:
                        area = "|area=all"
                    if i == 6:
                        max_dets = "|max_dets=1"
                    elif i == 7:
                        max_dets = "|max_dets=10"
                    else:
                        max_dets = "|max_dets=100"
                    figure = iou_metric + metric_name + iou_thresh + area + max_dets
                    samples.append(dl.PlotSample(figure=figure, legend="val", x=epoch, y=stat))
            self.model_entity.metrics.create(samples, dataset_id=self.model_entity.dataset_id)

        for epoch in range(num_epochs):
            logger.debug(f"Training epoch {epoch}")
            epoch_metrics = train_one_epoch(
                self.model,
                optimizer,
                data_loader,
                device=device,
                epoch=epoch,
                print_freq=10
                )
            epoch_end_callback(epoch_metrics, epoch)
            lr_scheduler.step()
            eval_results = evaluate(self.model, data_loader_test, device=device)
            eval_results_callback(eval_results, epoch)
        logger.info("Training finished successfully")

    def convert_from_dtlpy(self, data_path, **kwargs):
        input_size = self.configuration.get("input_size", 256)
        ##############
        # Validation #
        ##############

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError(
                'Couldnt find train set. Faster R-CNN requires train and validation set for training. '
                'Add a train set DQL filter in the dl.Model metadata'
                )
        if 'validation' not in subsets:
            raise ValueError(
                'Couldnt find validation set. Faster R-CNN requires train and validation set for training. '
                'Add a validation set DQL filter in the dl.Model metadata'
                )

        for subset in subsets:
            img_paths = glob(os.path.join(data_path, subset, 'items', '**'), recursive=True)
            for img_path in img_paths:
                img = Image.open(img_path)
                img.resize((input_size, input_size))
                img.save(img_path)
