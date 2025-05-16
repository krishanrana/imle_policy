import h5py
import numpy as np
import pickle as pkl
import pdb

# Path to the dataset
dataset_path = "../dataset/hdf5_datasets/kitchen/kitchen_perfect.hdf5"

# Initialize an empty dataset dictionary
dataset = {}

# Read the HDF5 dataset
with h5py.File(dataset_path, 'r') as f:
    actions = f['actions'][:]
    observations = f['observations'][:]
    terminals = f['terminals'][:]

# Create the dictionary of episodes
start_idx = 0  # Initialize the starting index for the first episode
episode_idx = 0  # Episode counter

for idx, is_terminal in enumerate(terminals):
    if is_terminal:  # Check if this index marks the end of an episode
        # Extract observations and actions for the current episode
        obs = observations[start_idx:idx + 1]  # Include the terminal step
        action = actions[start_idx:idx + 1]  # Include the terminal step
        
        # Add the episode data to the dataset dictionary
        dataset[episode_idx] = {
            "obs": obs,
            "action": action
        }
        
        # Update the starting index for the next episode
        start_idx = idx + 1
        episode_idx += 1  # Increment the episode counter


# save the dataset dictionary as a pickle file
out_file = '../dataset/hdf5_datasets/kitchen/kitchen_perfect_dataset.pkl'
with open(out_file, "wb") as f:
    pkl.dump(dataset, f)




