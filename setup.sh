#!/bin/bash

# Create environment name based on the exercise name
conda create -n 03-semantic-segmentation python=3.11 -y
conda activate 03-semantic-segmentation

pip install uv
uv pip install -r requirements.txt

python -m ipykernel install --user --name "03-semantic-segmentation"

# Build the notebooks
sh prepare-exercise.sh

# Download and extract data, etc.
gdown -O kaggle_data.zip 1ahuduC_4Ex84R7qKNRzAY6PiLRWX_J3I
unzip -u -qq kaggle_data.zip && rm kaggle_data.zip

# Return to base environment
conda deactivate
conda activate base
