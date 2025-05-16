#!/bin/bash

# Check if ~/.mujoco folder exists
if [ -d "$HOME/.mujoco" ]; then
    echo "~/.mujoco folder already exists. Exiting..."
    exit 1
fi

# Download MuJoCo to /tmp to avoid cluttering home directory
cd /tmp || exit 1
echo "Downloading MuJoCo..."
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz || {
    echo "Failed to download MuJoCo"
    exit 1
}

# Create .mujoco directory after successful download
echo "Creating ~/.mujoco directory..."
mkdir -p "$HOME/.mujoco" || {
    echo "Failed to create ~/.mujoco directory"
    rm mujoco210-linux-x86_64.tar.gz
    exit 1
}

# Extract it into the ~/.mujoco folder
echo "Extracting MuJoCo..."
tar -xzf mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco" || {
    echo "Failed to extract MuJoCo"
    rm mujoco210-linux-x86_64.tar.gz
    rmdir "$HOME/.mujoco"  # Remove the empty directory if extraction fails
    exit 1
}

# Clean up downloaded archive
rm mujoco210-linux-x86_64.tar.gz