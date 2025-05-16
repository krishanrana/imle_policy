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

        # Load demonstrations from the pickle file
        dataset_file = os.path.join(self.dataset_path)

        try:
            with open(dataset_file, 'rb') as f:
                self.rlds = pkl.load(f)
        except Exception as e:
            print(f"Failed to load pickle file: {e}")
            return
        
        # self.rlds = self.add_padding(self.rlds, self.obs_horizon-1, self.pred_horizon-1)

        # randomly sample 20% of the keys
        # ensure same shuffle order all runs
        np.random.seed(0)
        keys = list(self.rlds.keys())
        np.random.shuffle(keys)
        keys = keys[:int(dataset_percentage*len(keys))]
        self.rlds = {k: self.rlds[k] for k in keys}

        # Compute statistics for normalization
        self.stats = {'obs': None, 'action': None}
        self.compute_normalization_stats()

        # Create sample indices
        self.indices = self.create_sample_indices(self.rlds, sequence_length=self.pred_horizon)

        # Normalize the data
        self.normalize_rlds()

    def add_padding(self, rlds, pad_before, pad_after):
        # for padding before the first element of the episode we pad with the first element
        # for padding after the last element of the episode we pad with the last element

        for episode in rlds.keys():
            # Extract episode data
            obs = rlds[episode]['obs']
            action = rlds[episode]['action']
            
            # Pre-padding
            pre_pad_pos = [obs[0]] * pad_before
            pre_pad_action = [action[0]] * pad_before

            # Post-padding
            post_pad_pos = [obs[-1]] * pad_after
            post_pad_action = [action[-1]] * pad_after
            
            # Apply padding
            rlds[episode]['obs'] = np.array(pre_pad_pos + obs + post_pad_pos)
            rlds[episode]['action'] = np.array(pre_pad_action + action.tolist() + post_pad_action)
        
        return rlds

    def normalize_rlds(self):
        for episode in self.rlds.keys():
            self.rlds[episode]['obs'] = normalize_data(np.array(self.rlds[episode]['obs']), self.stats['obs'])
            self.rlds[episode]['action'] = normalize_data(np.array(self.rlds[episode]['action']), self.stats['action'])
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

        obs_data = np.concatenate([np.array(self.rlds[episode]['obs']) for episode in self.rlds.keys()], axis=0)
        action_data = np.concatenate([np.array(self.rlds[episode]['action']) for episode in self.rlds.keys()], axis=0)

        obs_stats = get_data_stats(obs_data)

        min_values = obs_stats["min"]
        max_values = obs_stats["max"]

        zero_variance_indices = np.where(min_values == max_values)

        min_values[zero_variance_indices] = -1
        max_values[zero_variance_indices] = 1

        obs_stats["min"] = min_values
        obs_stats["max"] = max_values

        self.stats['obs'] = obs_stats
        self.stats['action'] = get_data_stats(action_data)


    def sample_sequence(self, episode, buffer_start_idx, buffer_end_idx):
        obs = self.rlds[episode]['obs'][buffer_start_idx:buffer_start_idx+self.obs_horizon]
        action = self.rlds[episode]['action'][buffer_start_idx:buffer_end_idx]

        seq = {
            'obs': np.array(obs),
            'action': np.array(action),
        }
        return seq

    
    def __len__(self):
        return len(self.indices)
    
 
    def __getitem__(self, idx):
        episode, buffer_start_idx, buffer_end_idx = self.indices[idx]
        seq = self.sample_sequence(episode, buffer_start_idx, buffer_end_idx)

        obs = seq['obs']
        action = seq['action']

        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)


        return {
            'obs': obs,
            'action': action,
        }
    


if __name__ == '__main__':
    dataset = PolicyDataset(
                 dataset_path='../dataset/hdf5_datasets/kitchen/kitchen_dataset.pkl',
                 pred_horizon=16,
                 obs_horizon=2,
                 action_horizon=2,
                 dataset_percentage=1.0)
    
    idx=0
    while True:
        start_time = time.time()
        out = dataset.__getitem__(idx)
        pdb.set_trace()
        print(idx)
        # pdb.set_trace()
        # print(f"Time taken: {time.time() - start_time}")
        idx += 1
