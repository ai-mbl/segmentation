# Create environment name based on the exercise name
mamba create -n 05-image-segmentation python=3.11
mamba activate 05-image-segmentation
# Install additional requirements
mamba install -c pytorch pytorch jupyter imageio scipy tensorboard torchvision matplotlib cudatoolkit=10.2

# Return to base environment
mamba activate base

# Download and extract data, etc.
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp" -O kaggle_data.zip && rm -rf /tmp/cookies.txt
unzip -u -qq kaggle_data.zip && rm kaggle_data.zip