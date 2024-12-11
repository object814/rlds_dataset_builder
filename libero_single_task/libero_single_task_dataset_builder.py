from typing import Iterator, Tuple, Any
import os
import sys
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Add VLA_DIR to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '../../..')))
# Add LIBERO to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '../../LIBERO')))

from libero.libero import get_libero_path
from utils.LIBERO_utils import get_task_names

class LiberoSingleTask(tfds.core.GeneratorBasedBuilder):
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

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                    ),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        DATASET_BASE_PATH = get_libero_path("datasets")
        DATASET_PATH_DEMO = os.path.join(DATASET_BASE_PATH, self.dataset_name)
        task_names_demo = get_task_names(DATASET_PATH_DEMO)

        assert int(self.task_idx) == int(task_names_demo.index(self.task_name)), (
            f"Task index {self.task_idx} does not match task name {self.task_name}, "
            f"correct task index is {task_names_demo.index(self.task_name)}"
        )

        self.task_name_demo = task_names_demo[self.task_idx]

        episode_path = os.path.join(DATASET_PATH_DEMO, self.task_name_demo) + ".hdf5"

        return {
            'train': self._generate_examples([episode_path])
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        def _parse_example(episode_path, demo_key):
            with h5py.File(episode_path, "r") as F:
                if demo_key not in F['data']:
                    return None
                actions = F['data'][demo_key]["actions"][()]
                states = F['data'][demo_key]["obs"]["ee_states"][()]
                gripper_states = F['data'][demo_key]["obs"]["gripper_states"][()]
                joint_states = F['data'][demo_key]["obs"]["joint_states"][()]
                images = F['data'][demo_key]["obs"]["agentview_rgb"][()]
                wrist_images = F['data'][demo_key]["obs"]["eye_in_hand_rgb"][()]

            # Derive language instruction from filename
            raw_file_string = os.path.basename(episode_path)
            words = raw_file_string[:-10].split("_")
            command = ''
            for w in words:
                if "SCENE" in w:
                    command = ''
                    continue
                command = command + w + ' '
            command = command.strip()

            episode = []
            for i in range(actions.shape[0]):
                ep_state = np.concatenate((states[i], gripper_states[i]), axis=-1).astype(np.float32)
                ep_joint_state = joint_states[i].astype(np.float32)
                ep_image = images[i][::-1, ::-1]
                ep_wrist_image = wrist_images[i][::-1, ::-1]

                episode.append({
                    'observation': {
                        'image': ep_image,
                        'wrist_image': ep_wrist_image,
                        'state': ep_state,
                        'joint_state': ep_joint_state
                    },
                    'action': actions[i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == actions.shape[0] - 1),
                    'is_first': i == 0,
                    'is_last': i == (actions.shape[0] - 1),
                    'is_terminal': i == (actions.shape[0] - 1),
                    'language_instruction': command,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            return episode_path + f"_{demo_key}", sample

        for sample in paths:
            with h5py.File(sample, "r") as F:
                demo_keys = list(F['data'].keys())

            for demo_key in demo_keys:
                ret = _parse_example(sample, demo_key)
                if ret is not None:
                    yield ret
