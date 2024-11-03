from typing import Iterator, Tuple, Any
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import argparse

# Define paths
RLDS_SAVE_PATH = "/data/zhouhy/Datasets/LIBERO_spatial_segmented/task9_manual_segmented_rlds"
SEGMENTED_DATASET_PATH = "/data/zhouhy/Datasets/LIBERO_spatial_segmented/task9_manual_segment_raw/manual_segmented_dataset.pkl"

# Load the segmented dataset
with open(SEGMENTED_DATASET_PATH, 'rb') as f:
    segmented_dataset = pickle.load(f)

class LiberoSpatialManualSegment(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for libero_spatial segmented dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, segment_idx, *args, **kwargs):
        self.segment_idx = segment_idx  # Initialize segment_idx before calling super().__init__
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")  # Load the language embedding model

    @property
    def name(self):
        return f"libero_spatial_segment_{self.segment_idx}"

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of end effector 6D pose and gripper state (-1 to 1 for opening and closing)',
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
        segment_episodes = segmented_dataset[self.segment_idx]

        print(f"Segment #{self.segment_idx} has #{len(segment_episodes)} different segments")

        for idx, episode in enumerate(segment_episodes):
            language_instruction = episode['language_instruction']
            
            # Generate language embedding using the Universal Sentence Encoder
            language_embedding = self._embed([language_instruction])[0].numpy()

            # Update episode with the new language embedding
            episode['language_embedding'] = language_embedding
            
            # Yield each episode wrapped in a list
            yield f"segment_{self.segment_idx}_episode_{idx}", {
                'steps': [episode],  # Wrap the dictionary in a list
                'episode_metadata': {
                    'file_path': f"segment_{self.segment_idx}_episode_{idx}"
                }
            }

# Generate RLDS datasets for each segment
segment_num = 4
for segment_idx in range(segment_num):
    print(f"Building RLDS dataset for segment {segment_idx}...")
    builder = LiberoSpatialManualSegment(segment_idx=segment_idx)
    builder._data_dir = os.path.join(RLDS_SAVE_PATH, f"libero_spatial_manual_segmented_{segment_idx}")
    builder.download_and_prepare()
    print(f"RLDS dataset for segment {segment_idx} saved.")

print("All RLDS datasets have been generated and saved.")
