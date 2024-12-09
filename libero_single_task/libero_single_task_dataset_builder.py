from typing import Iterator, Tuple, Any

import os
import sys
# Add VLA_DIR to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '../../..')))
from utils.LIBERO_utils import get_task_names, extract_task_info
# Add LIBERO to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '../../LIBERO')))
from libero.libero import benchmark, get_libero_path

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

class LiberoSingleTask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for libero single task dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    @property
    def name(self):
        return f"{os.environ.get('DATASET_NAME')}_{int(os.environ.get('TASK_IDX'))}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = os.environ.get('TASK_NAME')
        self.task_idx = int(os.environ.get('TASK_IDX'))
        self.dataset_name = os.environ.get('DATASET_NAME')
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of end effector 6D pose and gripper state (-1 to 1 for opening and closing) ',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples()
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        FILTER_KEY = None
        VERBOSE = True

        DATASET_BASE_PATH = get_libero_path("datasets")
        DATASET_PATH_DEMO = os.path.join(DATASET_BASE_PATH, self.dataset_name)
        task_names_demo = get_task_names(DATASET_PATH_DEMO)
        
        assert int(self.task_idx) == int(task_names_demo.index(self.task_name)), f"Task index {self.task_idx} does not match task name {self.task_name}, correct task index is {task_names_demo.index(self.task_name)}"

        task_name_demo = task_names_demo[self.task_idx]
        [language_instruction, actions_batch, images_batch] = extract_task_info(
            DATASET_PATH_DEMO, task_name_demo, filter_key=FILTER_KEY, verbose=VERBOSE
        )

        for i in range(actions_batch.shape[0]):
            episode = []
            episode_length = actions_batch[i].shape[0]
            for j in range(episode_length):
                language_embedding = self._embed([language_instruction])[0].numpy()
                episode.append({
                    'observation': {
                        'image': images_batch[i][j],
                    },
                    'action': np.float32(actions_batch[i][j]),
                    'discount': 1.0,
                    'reward': float(j == (episode_length - 1)),
                    'is_first': j == 0,
                    'is_last': j == (episode_length - 1),
                    'is_terminal': j == (episode_length - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': os.path.join(DATASET_PATH_DEMO, task_name_demo)
                }
            }

            yield f"{task_name_demo}_{i}", sample
