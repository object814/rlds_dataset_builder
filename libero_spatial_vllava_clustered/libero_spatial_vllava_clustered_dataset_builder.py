import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import argparse
from typing import Iterator, Tuple, Any

# Parse arguments from command line
def parse_args():
    parser = argparse.ArgumentParser(description="Generate RLDS dataset for LIBERO clusters")
    parser.add_argument('--rlds_save_path', type=str, required=True, help="Base path for saving RLDS datasets")
    parser.add_argument('--clustered_dataset_path', type=str, required=True, help="Path to the clustered dataset")
    parser.add_argument('--tfds_data_dir', type=str, required=True, help="TFDS data directory for generated datasets")
    return parser.parse_args()

args = parse_args()

RLDS_SAVE_PATH = args.rlds_save_path
CLUSTERED_DATASET_PATH = args.clustered_dataset_path

with open(CLUSTERED_DATASET_PATH, 'rb') as f:
    clustered_dataset = pickle.load(f)

class LiberoSpatialCluster(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for libero_spatial clustered dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, cluster_idx, *args, **kwargs):
        self.cluster_idx = cluster_idx  # Initialize cluster_idx before calling super().__init__
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    @property
    def name(self):
        return f"libero_spatial_cluster_{self.cluster_idx}"

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
        cluster_episodes = clustered_dataset[self.cluster_idx]

        # Group neighboring timesteps into episodes
        grouped_episodes = []
        current_episode = []
        last_task_name_demo = None
        last_demo_idx = None
        last_segment_idx = None

        total_timesteps = len(cluster_episodes)
        episode_count = 0

        for episode_data in cluster_episodes:
            task_name_demo = episode_data['task_name_demo']
            demo_idx = episode_data['demo_idx']
            segment_idx = episode_data['segment_idx']
            observation = episode_data['observation']
            action = episode_data['action']
            language_instruction = episode_data['language_instruction']
            language_embedding = self._embed([language_instruction])[0].numpy()

            if (
                last_task_name_demo == task_name_demo and
                last_demo_idx == demo_idx and
                last_segment_idx is not None and
                segment_idx == last_segment_idx + 1
            ):
                # Continue the current episode
                current_episode.append({
                    'observation': {
                        'image': observation,
                    },
                    'action': np.float32(action),
                    'discount': 1.0,
                    'reward': 0.0,  # Set reward to 0 for intermediate steps
                    'is_first': False,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })
            else:
                # End the current episode and start a new one
                if current_episode:
                    current_episode[-1]['reward'] = 1.0  # Reward for the last step in the episode
                    current_episode[-1]['is_last'] = False
                    current_episode[-1]['is_terminal'] = False
                    grouped_episodes.append(current_episode)
                    episode_count += 1

                current_episode = [{
                    'observation': {
                        'image': observation,
                    },
                    'action': np.float32(action),
                    'discount': 1.0,
                    'reward': 0.0,  # Set reward to 0 initially
                    'is_first': False,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }]

            last_task_name_demo = task_name_demo
            last_demo_idx = demo_idx
            last_segment_idx = segment_idx

        # Handle the last episode
        if current_episode:
            current_episode[-1]['reward'] = 1.0  # Reward for the last step in the episode
            current_episode[-1]['is_last'] = False
            current_episode[-1]['is_terminal'] = False
            grouped_episodes.append(current_episode)
            episode_count += 1

        # Output the number of timesteps and episodes for this cluster
        print(f"Cluster {self.cluster_idx}: {total_timesteps} timesteps, {episode_count} episodes")

        # Yield each grouped episode
        for episode_idx, episode in enumerate(grouped_episodes):
            yield f"cluster_{self.cluster_idx}_episode_{episode_idx}", {
                'steps': episode,
                'episode_metadata': {
                    'file_path': f"cluster_{self.cluster_idx}_episode_{episode_idx}"
                }
            }

# Generate RLDS datasets for each cluster
for cluster_idx in clustered_dataset.keys():
    print(f"Building RLDS dataset for cluster {cluster_idx}...")
    builder = LiberoSpatialCluster(cluster_idx=cluster_idx)
    builder._data_dir = os.path.join(RLDS_SAVE_PATH, f"libero_spatial_cluster_{cluster_idx}")
    builder.download_and_prepare()
    print(f"RLDS dataset for cluster {cluster_idx} saved.")

print("All RLDS datasets have been generated and saved.")
