import json
import os
import torch
import torchvision
import torchaudio
import dtlpy as dl
import torch.nn.functional
import torch.nn
import logging
import soundfile
from speechbrain.inference import EncoderClassifier

logger = logging.getLogger('EncoderClassifier-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for SpeechBrain Encoder Classifier model',
                              init_inputs={'model_entity': dl.Model})
class EncoderClassifierAdapter(dl.BaseModelAdapter):
    """
    SpeechBrain Encoder Classifier Model adapter using Pytorch.
    The class bind Dataloop model and model entities with model code implementation
    """

    def __init__(self, model_entity: dl.Model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        # Load languages from JSON file
        json_path = os.path.join(os.getcwd(), 'languages_labels.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.languages_list = data.get('languages', [])
        if not self.languages_list:
            raise Exception("Languages list is empty or not found in JSON file.")
        self.model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                                    savedir="pretrained_models/lang-id-voxlingua107-ecapa")
        logger.info(f"Loaded model from library successfully")
        self.model.to(self.device)
        self.model.eval()

    def prepare_item_func(self, item):
        return item

    def save(self, local_path, **kwargs):
        """ Saves configuration and weights locally

            The function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', 'model.pth')
        torch.save(self.model, os.path.join(local_path, weights_filename))
        self.configuration['weights_filename'] = weights_filename

    def predict(self, batch: [dl.Item], **kwargs):
        """ Model inference (predictions) on batch of audio files

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        logger.info('Encoder Classifier prediction started')
        batch_annotations = list()
        for item in batch:
            filename = item.download(overwrite=True)
            logger.info(f'Encoder Classifier predicting {filename}, started.')
            # Get the format from filename and adding it to torchaudio load
            signal = self.model.load_audio(filename)
            prediction = self.model(signal)
            logger.info(f'Encoder Classifier predicting {filename}, done.')

            # Convert log-likelihoods to linear-scale likelihoods
            log_likelihoods = prediction[0][0]
            linear_likelihoods = log_likelihoods.exp()

            # Find the indices of the languages with the highest likelihoods
            sorted_indices = linear_likelihoods.argsort(descending=True)

            # Initialize list for languages with confidence > 30%
            best_languages = set()
            confidences = set()

            # Calculate the total sum of linear likelihoods
            total_likelihood = linear_likelihoods.sum()

            # Check confidence for the top 3 languages
            for idx in sorted_indices[:3]:
                confidence = linear_likelihoods[int(idx)] / total_likelihood
                if float(confidence) > 0.30:
                    confidences.add(confidence)
                    best_languages.add(self.languages_list[idx])
            best_languages_list = list(best_languages)
            confidences_list = list(confidences)

            if len(best_languages_list) == 0:
                best_language_index = linear_likelihoods.argmax()
                confidence = linear_likelihoods[int(best_language_index)] / total_likelihood
                confidences_list.append(confidence)
                best_languages_list.append(self.languages_list[best_language_index])

            collection = dl.AnnotationCollection()
            for label, confidence in zip(best_languages_list, confidences_list):
                collection.add(annotation_definition=dl.Classification(label=label),
                               model_info={'name': self.model_entity.name,
                                           'confidence': confidence,
                                           'model_id': self.model_entity.id})
                logger.debug(f"Predicted {label} with confidence {round(float(confidence), 2)}.")
                batch_annotations.append(collection)
        return batch_annotations
