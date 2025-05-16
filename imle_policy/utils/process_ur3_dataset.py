import os
import numpy as np
import pickle as pkl
import pdb

# Define the function for preprocessing the dataset
def preprocess_dataset(dataset_path, output_file=None):
    """
    Preprocess the trajectory dataset to create a dictionary organized by episodes.

    Args:
        dataset_path (str): Path to the dataset directory containing .npy files.
        output_file (str): Optional path to save the preprocessed dataset as a .pkl file.

    Returns:
        dict: Preprocessed dataset organized by episodes.
    """
    # Paths to the .npy files
    action_npy_file = os.path.join(dataset_path, 'data_act.npy')
    data_obs_npy_file = os.path.join(dataset_path, 'data_obs.npy')
    mask_npy_file = os.path.join(dataset_path, 'data_msk.npy')

    # Load data from .npy files
    actions = np.load(action_npy_file)  # Shape: [600, 1000, 2]
    data_obs = np.load(data_obs_npy_file)  # Shape: [600, 1000, 6]
    masks = np.load(mask_npy_file)  # Shape: [600, 1000, 1]

    # Reshape masks to match the first two dimensions for slicing
    masks = masks.squeeze(-1)  # Shape: [600, 1000]

    # Initialize a dictionary to store the preprocessed dataset
    dataset = {}

    # Iterate over each episode
    num_episodes = masks.shape[0]  # 600 episodes
    for episode in range(num_episodes):
        # Extract valid timesteps for the current episode
        valid_mask = masks[episode] > 0  # Shape: [1000]

        # Extract valid observations and actions
        valid_obs = data_obs[episode][valid_mask, :]  # Valid observations for this episode
        valid_actions = actions[episode][valid_mask, :]  # Valid actions for this episode


        # subsample the data to extract every 2nd timestep
        valid_obs = valid_obs[::5]
        valid_actions = valid_actions[::5]

        # Store the valid data in the dataset dictionary
        dataset[episode] = {
            "obs": valid_obs,
            "action": valid_actions,
        }

    # Optionally save the preprocessed dataset as a .pkl file
    if output_file:
        with open(output_file, "wb") as f:
            pkl.dump(dataset, f)
        print(f"Preprocessed dataset saved to {output_file}")

    print("Preprocessing complete.")

    return dataset

# Example usage
dataset_path = "../dataset/vqbet_datasets/ur3"  # Replace with the actual path to your dataset
output_file = "../dataset/vqbet_datasets/ur3/ur3_blockpush_subsampled_5.pkl"  # Output file
dataset = preprocess_dataset(dataset_path, output_file=output_file)
