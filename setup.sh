#!/bin/bash

# Create environment name based on the exercise name
conda create -n 03-segmentation python=3.11 -y
conda activate 03-segmentation

pip install uv
uv pip install -r requirements.txt

python -m ipykernel install --user --name "03-segmentation"

# Build the notebooks
sh prepare-exercise.sh

# Download and extract data, etc.
echo -e "\n downloading data...\n"
gdown -O kaggle_data.zip 1ahuduC_4Ex84R7qKNRzAY6PiLRWX_J3I
unzip -u -qq kaggle_data.zip && rm kaggle_data.zip

aws s3 cp "s3://dl-at-mbl-2023-data/woodshole_new.zip" "." --no-sign-request
unzip woodshole_new.zip
mkdir tissuenet_data
mv woodshole_new/* tissuenet_data
rm woodshole_new.zip
rm -r woodshole_new

# Return to base environment
conda deactivate
conda activate base
