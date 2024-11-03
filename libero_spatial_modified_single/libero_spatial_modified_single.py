from typing import Iterator, Tuple, Any
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from pathlib import Path

os.environ['TFDS_DATA_DIR'] = '/data/zhouhy/Datasets/modified_libero_rlds_single'

# Define LiberoSpatialModified class to generate datasets from the .pkl file
class LiberoSpatialModified(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for modified_libero_spatial dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with modified dataset.',
    }

    @property
    def name(self):
        return f"libero_spatial_modified_{os.environ.get('TASK_IDX')}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = os.environ.get('TASK_NAME')
        self.task_idx = int(os.environ.get('TASK_IDX'))
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        # Load the modified .pkl dataset
        self.pkl_file_path = '/data/zhouhy/Datasets/modified_libero_rlds_pkl/libero_spatial_no_noops.pkl'
        with open(self.pkl_file_path, 'rb') as f:
            self.dataset = pickle.load(f)
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_primary': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation, resized to 256x256.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, 6D pose and gripper state.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount (defaults to 1).'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward for the final step of each demo.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on the first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on the last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on the terminal step.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction for the task.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Language embedding using Universal Sentence Encoder.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='File path of the episode data.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples()
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split using the provided .pkl file."""
        
        # Get the relevant task from the .pkl dataset
        assert self.task_name in self.dataset, f"Task '{self.task_name}' not found in the dataset."

        data = self.dataset[self.task_name]
        actions_batch = np.array(data['actions'])
        images_batch = np.array(data['observations'])

        for i in range(actions_batch.shape[0]):
            episode = []
            episode_length = actions_batch[i].shape[0]
            for j in range(episode_length):
                # Create the language embedding using Universal Sentence Encoder
                language_instruction = self.task_name
                language_embedding = self._embed([language_instruction])[0].numpy()

                episode.append({
                    'observation': {
                        'image': np.uint8(images_batch[i][j]),  # Resize to 256x256 handled externally
                    },
                    'action': np.float32(actions_batch[i][j]),
                    'discount': 1.0,
                    'reward': 0,
                    'is_first': False,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': f"/data/zhouhy/Datasets/modified_libero_rlds_single/{self.task_name}"
                }
            }

            yield f"{self.task_name}_{i}", sample
