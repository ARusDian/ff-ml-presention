#!/bin/bash

# Define environment name and Python version
ENV_NAME="myenv"
PYTHON_VERSION="3.10"

# Function to check the success of the last command and exit if it failed
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Download and install Anaconda
echo "Downloading Anaconda..."
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
check_success "Failed to download Anaconda."

echo "Installing Anaconda..."
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p $HOME/anaconda3
check_success "Failed to install Anaconda."
rm Anaconda3-2024.10-1-Linux-x86_64.sh
export PATH="$HOME/anaconda3/bin:$PATH"
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

# Create a new conda environment with Python 3.10
echo "Creating conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
check_success "Failed to create conda environment."

# Activate the environment
echo "Activating conda environment..."
source activate $ENV_NAME
check_success "Failed to activate conda environment."

# Install torch, torchvision, and torchaudio with CUDA support
echo "Installing torch, torchvision, and torchaudio with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
check_success "Failed to install torch, torchvision, and torchaudio."

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
check_success "Failed to install dependencies from requirements.txt."


# Set the environment as the default in .bashrc
echo "Setting the environment as default in .bashrc..."
echo "conda activate $ENV_NAME" >> ~/.bashrc

# Inform the user
echo "Installation complete. The environment '$ENV_NAME' with Python $PYTHON_VERSION has been created and set as default in .bashrc."

# Verify installation
echo "Verifying installation..."
conda list
check_success "Failed to verify conda environment installation."

echo "All steps completed successfully."
