import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
import pdb
import time
import torchvision.transforms as transforms
import torchvision
import pickle as pkl
from imle_policy.utils.rotation_transforms import quaternion_to_rotation_6D, rotation_6D_to_quaternion
import h5py


# Helper function to get normalization stats
def get_data_stats(data):
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

# Normalize data to [-1, 1]
def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])  # Normalize to [0,1]
    ndata = ndata * 2 - 1  # Normalize to [-1,1]
    return ndata

# Unnormalize data
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class PolicyDataset(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 dataset_percentage: float = 1.0):
        
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # extract task name from the dataset path
        self.task_name = self.dataset_path.split('/')[-1]
        self.task_name = self.task_name.split('_')[0]

        # Load demonstrations from the pickle file
        dataset_file = os.path.join(self.dataset_path)

        self.raw_data = h5py.File(dataset_file, 'r')
        self.rlds = self.create_rlds()


        # create a new entry to rlds for each episode to store position data which is a slice of the obs
        # convert quaternion to rotation 6D for both obs and action

        if self.task_name == 'transport':
            for episode in self.rlds.keys():
                pos_array_0 = self.rlds[episode]['obs']['robot0_eef_pos']
                quat_array_0 = self.rlds[episode]['obs']['robot0_eef_quat']
                quat_array_0 = np.array([quaternion_to_rotation_6D(q) for q in quat_array_0])
                gripper_array_0 = self.rlds[episode]['obs']['robot0_gripper_qpos']

                pos_array_1 = self.rlds[episode]['obs']['robot1_eef_pos']
                quat_array_1 = self.rlds[episode]['obs']['robot1_eef_quat']
                quat_array_1 = np.array([quaternion_to_rotation_6D(q) for q in quat_array_1])
                gripper_array_1 = self.rlds[episode]['obs']['robot1_gripper_qpos']

                obs = np.concatenate([pos_array_0, quat_array_0, gripper_array_0, pos_array_1, quat_array_1, gripper_array_1], axis=1)
                self.rlds[episode]['obs'] = obs
        else:
            for episode in self.rlds.keys():   
                pos_array = self.rlds[episode]['obs']['robot0_eef_pos']
                quat_array = self.rlds[episode]['obs']['robot0_eef_quat']
                quat_array = np.array([quaternion_to_rotation_6D(q) for q in quat_array])
                gripper_array = self.rlds[episode]['obs']['robot0_gripper_qpos']

                obs = np.concatenate([pos_array, quat_array, gripper_array], axis=1)
                self.rlds[episode]['obs'] = obs
                    
        # self.rlds = self.add_padding(self.rlds, self.obs_horizon-1, self.pred_horizon-1)

        # randomly sample 20% of the keys
        # ensure same shuffle order all runs
        np.random.seed(0)
        keys = list(self.rlds.keys())
        np.random.shuffle(keys)
        keys = keys[:int(dataset_percentage*len(keys))]
        self.rlds = {k: self.rlds[k] for k in keys}

        # Compute statistics for normalization
        self.stats = {}
        self.compute_normalization_stats()

        # Create sample indices
        self.indices = self.create_sample_indices(self.rlds, sequence_length=self.pred_horizon)

        # Normalize the data
        self.normalize_rlds()

    def create_rlds(self):

        # Get the list of demonstrations
        demos = list(self.raw_data["data"].keys())

        # Sort demonstrations numerically
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        # Initialize a dictionary to store all demonstrations
        all_demos = {}

        # Iterate through each demonstration
        for demo_idx, ep in enumerate(demos):
            demo_grp = self.raw_data["data/{}".format(ep)]
            # Extract actions
            actions = demo_grp["actions"][:]
            # Extract observations
            obs = {k: demo_grp[f"obs/{k}"][:] for k in demo_grp["obs"]}
            # Reshape or process images (if applicable)
            # Remove images from the observation

            if 'agentview_image' in obs:
                obs.pop('agentview_image')
            if 'sideview_image' in obs:
                obs.pop('sideview_image')
            if 'robot0_eye_in_hand_image' in obs:
                obs.pop('robot0_eye_in_hand_image')
            if 'robot1_eye_in_hand_image' in obs:
                obs.pop('robot1_eye_in_hand_image')

            # Store the demonstration in the dictionary
            all_demos[demo_idx] = {
                'obs': obs,
                'action': actions,
            }

        return all_demos
            

    def add_padding(self, rlds, pad_before, pad_after):
        # for padding before the first element of the episode we pad with the first element
        # for padding after the last element of the episode we pad with the last element

        for episode in rlds.keys():
            # Extract episode data
            obs = rlds[episode]['obs']
            action = rlds[episode]['action']
            fimages = rlds[episode]['front_images']
            gimages = rlds[episode]['hand_images']

            
            # Pre-padding
            pre_pad_obs = [obs[0].tolist()] * pad_before
            pre_pad_action = [action[0].tolist()] * pad_before
            pre_pad_fimages = np.repeat(fimages[:1], pad_before, axis=0)
            pre_pad_gimages = np.repeat(gimages[:1], pad_before, axis=0)

            
            # Post-padding
            post_pad_obs = [obs[-1].tolist()] * pad_after
            post_pad_action = [action[-1].tolist()] * pad_after
            post_pad_fimages = np.repeat(fimages[-1:], pad_after, axis=0)
            post_pad_gimages = np.repeat(gimages[-1:], pad_after, axis=0)
            
            # Apply padding
            rlds[episode]['action'] = np.array(pre_pad_action + action.tolist() + post_pad_action)
            rlds[episode]['obs'] = np.array(pre_pad_obs + obs.tolist() + post_pad_obs)
            rlds[episode]['front_images'] = np.concatenate([pre_pad_fimages, fimages, post_pad_fimages], axis=0)
            rlds[episode]['hand_images'] = np.concatenate([pre_pad_gimages, gimages, post_pad_gimages], axis=0)

        
        return rlds

    def normalize_rlds(self):
        for episode in self.rlds.keys():
            self.rlds[episode]['action'] = normalize_data(np.array(self.rlds[episode]['action']), self.stats['action'])
            self.rlds[episode]['obs'] = normalize_data(np.array(self.rlds[episode]['obs']), self.stats['obs'])
            # self.rlds[episode]['front_images'] = self.rlds[episode]['front_images'] / 255.0
            # self.rlds[episode]['front_images'] = np.moveaxis(self.rlds[episode]['front_images'], -1, 1)
            # self.rlds[episode]['hand_images'] = self.rlds[episode]['hand_images'] / 255.0
            # self.rlds[episode]['hand_images'] = np.moveaxis(self.rlds[episode]['hand_images'], -1, 1)
        return

    def create_sample_indices(self, rlds_dataset, sequence_length=16):
        indices = []
        for episode in rlds_dataset.keys():
            episode_length = len(rlds_dataset[episode]['obs'])
            range_idx = episode_length - (sequence_length)
            for idx in range(range_idx):
                buffer_start_idx = idx
                buffer_end_idx = idx + sequence_length
                indices.append([episode, buffer_start_idx, buffer_end_idx])
        indices = np.array(indices)

        return indices

    def compute_normalization_stats(self):

        action_data = np.concatenate([np.array(self.rlds[episode]['action']) for episode in self.rlds.keys()], axis=0)
        obs_data = np.concatenate([np.array(self.rlds[episode]['obs']) for episode in self.rlds.keys()], axis=0)
        self.stats['action'] = get_data_stats(action_data)
        self.stats['obs'] = get_data_stats(obs_data)
  


    def sample_sequence(self, episode, buffer_start_idx, buffer_end_idx):
        obs = self.rlds[episode]['obs'][buffer_start_idx:buffer_start_idx+self.obs_horizon]
        action = self.rlds[episode]['action'][buffer_start_idx:buffer_end_idx]

        # sample images from raw data
        ep_name = 'demo_{}'.format(episode)
        ep_data = self.raw_data["data/{}".format(ep_name)]

        if self.task_name == 'toolhang':
            fimages = ep_data["obs/sideview_image"][buffer_start_idx:buffer_start_idx+self.obs_horizon]
            gimages = ep_data["obs/robot0_eye_in_hand_image"][buffer_start_idx:buffer_start_idx+self.obs_horizon]

            fimages = np.moveaxis((fimages / 255.0), -1, 1)
            gimages = np.moveaxis((gimages / 255.0), -1, 1)

            seq = {
                    'obs': np.array(obs),
                    'action': np.array(action),
                    'front_images': np.array(fimages),
                    'hand_images': np.array(gimages)
                }

        elif self.task_name == 'transport':
            fimages_0 = ep_data["obs/shouldercamera0_image"][buffer_start_idx:buffer_start_idx+self.obs_horizon]
            gimages_0 = ep_data["obs/robot0_eye_in_hand_image"][buffer_start_idx:buffer_start_idx+self.obs_horizon]
            fimages_1 = ep_data["obs/shouldercamera1_image"][buffer_start_idx:buffer_start_idx+self.obs_horizon]
            gimages_1 = ep_data["obs/robot1_eye_in_hand_image"][buffer_start_idx:buffer_start_idx+self.obs_horizon]

            fimages_0 = np.moveaxis((fimages_0 / 255.0), -1, 1)
            gimages_0 = np.moveaxis((gimages_0 / 255.0), -1, 1)
            fimages_1 = np.moveaxis((fimages_1 / 255.0), -1, 1)
            gimages_1 = np.moveaxis((gimages_1 / 255.0), -1, 1)

            seq = {
                    'obs': np.array(obs),
                    'action': np.array(action),
                    'front_images_0': np.array(fimages_0),
                    'hand_images_0': np.array(gimages_0),
                    'front_images_1': np.array(fimages_1),
                    'hand_images_1': np.array(gimages_1)
                }

        return seq

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        episode, buffer_start_idx, buffer_end_idx = self.indices[idx]
        seq = self.sample_sequence(episode, buffer_start_idx, buffer_end_idx)

        obs = seq['obs']
        action = seq['action']
        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        
        if self.task_name == 'transport':
            fimages_0 = seq['front_images_0']
            gimages_0 = seq['hand_images_0']
            fimages_1 = seq['front_images_1']
            gimages_1 = seq['hand_images_1']
            fimages_0 = torch.tensor(fimages_0, dtype=torch.float32)
            gimages_0 = torch.tensor(gimages_0, dtype=torch.float32)
            fimages_1 = torch.tensor(fimages_1, dtype=torch.float32)
            gimages_1 = torch.tensor(gimages_1, dtype=torch.float32)

    

            data = {
                    'obs': obs,
                    'action': action,
                    'front_images_0': fimages_0,
                    'hand_images_0': gimages_0,
                    'front_images_1': fimages_1,
                    'hand_images_1': gimages_1
                }

        else:
            fimages = seq['front_images']
            gimages = seq['hand_images']
            fimages = torch.tensor(fimages, dtype=torch.float32)
            gimages = torch.tensor(gimages, dtype=torch.float32)

            data = {
                    'obs': obs,
                    'action': action,
                    'front_images': fimages,
                    'hand_images': gimages
                }

        return data
    


if __name__ == '__main__':
    dataset = PolicyDataset(
                 dataset_path='dataset/hdf5_datasets/transport/ph/image_v141.hdf5',
                 pred_horizon=16,
                 obs_horizon=2,
                 action_horizon=2,
                 dataset_percentage=1.0)
    
    idx=0
    while True:
        start_time = time.time()
        temp = dataset.__getitem__(idx)
        print(idx)
        pdb.set_trace()
        # print(f"Time taken: {time.time() - start_time}")
        idx += 1
