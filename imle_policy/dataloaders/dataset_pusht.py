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

        self.transform = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.RandomCrop((216, 288)),
                         transforms.ToTensor()])

        # Load demonstrations from the pickle file

        # pdb.set_trace()

        dataset_file = os.path.join(self.dataset_path)
        try:
            with open(dataset_file, 'rb') as f:
                self.rlds = pkl.load(f)
        except Exception as e:
            print(f"Failed to load pickle file: {e}")
            return
        
        # create a new entry to rlds for each episode to store position data which is a slice of the obs
        for episode in self.rlds.keys():
            pos_array = []
            for i in range(len(self.rlds[episode]['obs'])):
                pos_array.append(self.rlds[episode]['obs'][i][:2])
            
            self.rlds[episode]['agent_pos'] = pos_array


        self.rlds = self.add_padding(self.rlds, self.obs_horizon-1, self.pred_horizon-1)

        # randomly sample 20% of the keys
        # ensure same shuffle order all runs
        np.random.seed(0)
        keys = list(self.rlds.keys())
        np.random.shuffle(keys)
        keys = keys[:int(dataset_percentage*len(keys))]
        self.rlds = {k: self.rlds[k] for k in keys}

        # Compute statistics for normalization
        self.stats = {'agent_pos': None, 'action': None}
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
            agent_pos = rlds[episode]['agent_pos']
            action = rlds[episode]['action']
            images = rlds[episode]['images']
            
            # Pre-padding
            pre_pad_pos = [agent_pos[0]] * pad_before
            pre_pad_action = [action[0]] * pad_before
            pre_pad_images = np.repeat(images[:1], pad_before, axis=0)

            
            # Post-padding
            post_pad_pos = [agent_pos[-1]] * pad_after
            post_pad_action = [action[-1]] * pad_after
            post_pad_images = np.repeat(images[-1:], pad_after, axis=0)
            
            # Apply padding
            rlds[episode]['agent_pos'] = np.array(pre_pad_pos + agent_pos + post_pad_pos)
            rlds[episode]['action'] = np.array(pre_pad_action + action.tolist() + post_pad_action)
            rlds[episode]['images'] = np.concatenate([pre_pad_images, images, post_pad_images], axis=0)
        
        return rlds

    def normalize_rlds(self):
        for episode in self.rlds.keys():
            self.rlds[episode]['agent_pos'] = normalize_data(np.array(self.rlds[episode]['agent_pos']), self.stats['agent_pos'])
            self.rlds[episode]['action'] = normalize_data(np.array(self.rlds[episode]['action']), self.stats['action'])
            self.rlds[episode]['images'] = self.rlds[episode]['images'] / 255.0
            self.rlds[episode]['images'] = np.moveaxis(self.rlds[episode]['images'], -1, 1)
        return

    def create_sample_indices(self, rlds_dataset, sequence_length=16):
        indices = []
        for episode in rlds_dataset.keys():
            episode_length = len(rlds_dataset[episode]['agent_pos'])
            range_idx = episode_length - (sequence_length)
            for idx in range(range_idx):
                buffer_start_idx = idx
                buffer_end_idx = idx + sequence_length
                indices.append([episode, buffer_start_idx, buffer_end_idx])
        indices = np.array(indices)

        return indices

    def compute_normalization_stats(self):

        agent_pos_data = np.concatenate([np.array(self.rlds[episode]['agent_pos']) for episode in self.rlds.keys()], axis=0)
        action_data = np.concatenate([np.array(self.rlds[episode]['action']) for episode in self.rlds.keys()], axis=0)

        self.stats['agent_pos'] = get_data_stats(agent_pos_data)
        self.stats['action'] = get_data_stats(action_data)


    def sample_sequence(self, episode, buffer_start_idx, buffer_end_idx):
        agent_pos = self.rlds[episode]['agent_pos'][buffer_start_idx:buffer_start_idx+self.obs_horizon]
        frames = self.rlds[episode]['images'][buffer_start_idx:buffer_start_idx+self.obs_horizon]
        action = self.rlds[episode]['action'][buffer_start_idx:buffer_end_idx]
        # action = self.rlds[episode]['agent_pos'][buffer_start_idx+1:buffer_end_idx+1]

        seq = {
            'agent_pos': np.array(agent_pos),
            'action': np.array(action),
            'frames': np.array(frames)
        }
        return seq

    
    def __len__(self):
        return len(self.indices)
    
    def visualize_images_in_row(self, tensor):
        # Ensure the input tensor is in the right shape
        assert tensor.shape == (2, 3, 96, 96), "Tensor should have shape (16, 3, 216, 288)"
        
        # Create a grid of images in a single row
        grid_img = torchvision.utils.make_grid(tensor, nrow=16)  # Arrange 16 images in a single row      
        # Convert the tensor to a numpy array for displaying
        plt.figure(figsize=(20, 5))  # Adjust figure size if necessary
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())  # Permute to get (H, W, C) for display
        plt.axis('off')  # Hide axis
        plt.show()

    def __getitem__(self, idx):
        episode, buffer_start_idx, buffer_end_idx = self.indices[idx]
        seq = self.sample_sequence(episode, buffer_start_idx, buffer_end_idx)

        agent_pos = seq['agent_pos']
        action = seq['action']
        frames = seq['frames']

        # Convert to tensors
        agent_pos = torch.tensor(agent_pos, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        frames = torch.tensor(frames, dtype=torch.float32)

        # self.visualize_images_in_row(torch.tensor(frames))

        return {
            'agent_pos': agent_pos,
            'action': action,
            'image': frames,
        }
    


if __name__ == '__main__':
    dataset = PolicyDataset(
                 dataset_path='dataset/pusht.pkl',
                 pred_horizon=16,
                 obs_horizon=2,
                 action_horizon=2,
                 dataset_percentage=1.0)
    
    idx=0
    while True:
        start_time = time.time()
        dataset.__getitem__(idx)
        pdb.set_trace()
        print(idx)
        # pdb.set_trace()
        # print(f"Time taken: {time.time() - start_time}")
        idx += 1
