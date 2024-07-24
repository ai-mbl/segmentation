#!/bin/bash

# Create environment name based on the exercise name
conda create -n 03-semantic-segmentation python=3.10 -y
conda activate 03-semantic-segmentation
# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "03-semantic-segmentation" ]]; then
    echo "Environment activated successfully for package installs"
    conda install pytorch pytorch-cuda=11.8 jupyter imageio scipy scikit-image tensorboard torchvision matplotlib ipykernel jupytext -c pytorch -c nvidia -y
    python -m ipykernel install --user --name "03-semantic-segmentation"
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
fi

# Build the notebooks
sh prepare-exercise.sh

# Return to base environment
conda deactivate
conda activate base

# Download and extract data, etc.
gdown -O kaggle_data.zip 1ahuduC_4Ex84R7qKNRzAY6PiLRWX_J3I
unzip -u -qq kaggle_data.zip && rm kaggle_data.zip
