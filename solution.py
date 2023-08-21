# %% [markdown]
# # Exercise 05: Image Segmentation
#
# <hr style="height:2px;">
#
# In this notebook, we will train a 2D U-Net for nuclei segmentation in the Kaggle Nuclei dataset.
#
# Written by Valentyna Zinchenko, Constantin Pape and William Patton.

# %% [markdown]
# <div class="alert alert-danger">
# Please use kernel <code>05-image-segmentation</code> for this exercise.
# </div>

# %% [markdown]
# Our goal is to produce a model that can take an image as input and produce a segmentation as shown in this table.
#
# | Image | Mask |
# | :-: | :-: |
# | ![image](static/image.png) | ![mask](static/mask.png) |

# %% [markdown]
# <hr style="height:2px;">
#
# ## The libraries

# %%
%matplotlib inline
%load_ext tensorboard
import os
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision import transforms

# %% [markdown]
# <hr style="height:2px;">
#
# ## Data Loading and Preprocessing

# %% [markdown]
# ### Data exploration
# For this exercise we will be using the Kaggle 2018 Data Science Bowl data.
# We will try to segment it with a state of the art network.


# %% [markdown]
# Make sure that the data was successfully extracted:
# if everything went fine, you should have folders `nuclei_train_data` and `nuclei_val_data`
# in your working directory. Let's check if it is the case:

# %%
# list all of the directories
list([x for x in Path().iterdir() if x.is_dir()])

# %% [markdown]

# <div class="alert alert-block alert-info">
#     <p><b>Task 1.1: </b>: Explore the contents of both folders. Running `ls your_folder_name`
#     should display you what is stored in the folder of your interest.</p>
#     </p>You should be familiar with how the images are stored and the storage format.</p>
#     <b>Questions:</b>
#     <ol>
#         <li>How many image/mask pairs are there in the training/validation set?</li>
#         <li>What is the file type of the images/masks?</li>
#     </ol>
# </div>


# %%
# Write your answers here:
num_train_pairs: int = ...
num_val_pairs: int = ...
image_file_type: str = ".***"
mask_file_type: str = ".***"

# %% tags=["solution"]
# Write your answers here:
num_train_pairs = 536
num_val_pairs = 134
image_file_type = ".tif"
mask_file_type = ".tif"

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p><b>Task 1.2: </b>: Visualize the image associated with the following mask:</p>
#     <p>Hint: you can use the following function to display an image:</p>
# </div>


# %%
def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)


# %%
# show mask
show_one_image(
    Path(
        "nuclei_train_data/f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81/mask.tif"
    )
)

# %%
# show image
show_one_image(Path(...))

# %% tags=["solution"]
# show image
show_one_image(
    Path(
        "nuclei_train_data/f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81/image.tif"
    )
)

# %% [markdown]
# ### Data Processing
# Making the data accessible for training. What one would normally start with in any machine learning pipeline is writing a dataset - a class that will fetch the training samples. Once you switch to using your own data, you would have to figure out how to fetch the data yourself. Luckily most of the functionality is already provided by PyTorch, but what you need to do is to write a class, that will actually supply the dataloader with training samples - a Dataset.
#
# Torch Dataset docs can be found [here](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) and a totorial on how to use them can be found [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class).
#
# The main idea: any Dataset class should implement two "magic" methods:
# 1) `__len__(self)`: this defines the behavior of the python builtin `len` function. i.e:
#     `len(dataset)` => number of elements in your dataset
# 2) `__getitem__(self, idx)`: this defines bracket indexing behavior of your class. i.e:
#     `dataset[idx]` => element of your dataset associated with `idx`
#
# For this exercise you will not have to do it yourself yet, but please carefully read through the provided class:


# %%
# any PyTorch dataset class should inherit the initial torch.utils.data.Dataset
class NucleiDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),  # some of the images are RGB
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), # 0.5 = mean and 0.5 = variance 
            ]
        )

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = Image.open(img_path)
        image = self.inp_transforms(image)
        mask_path = os.path.join(self.root_dir, self.samples[idx], "mask.tif")
        mask = transforms.ToTensor()(Image.open(mask_path))
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p><b>Task 1.3</b>: Use the defined dataset to show a random image/mask pair</p>
#     <p>Hint: use the <code>len</code> function and <code>[]</code> indexing defined by <code>__len__</code> and
#     <code>__get_index__</code> in the Dataset class to fill in the function below.</p>
# </div>


# %%
def show_random_dataset_image(dataset):
    idx = ...
    img, mask = ...
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


# %% tags=["solution"]
def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


# %%
TRAIN_DATA_PATH = "nuclei_train_data"
train_data = NucleiDataset(TRAIN_DATA_PATH)

show_random_dataset_image(train_data)

# %% [markdown]
#
#
# As you can probably see, if you clicked enough times, some of the images are really huge! What happens if we load them into memory and run the model on them? We might run out of memory. That's why normally, when training networks on images or volumes one has to be really careful about the sizes. In practice, you would want to regulate their size. Additional reason for restraining the size is: if we want to train in batches (faster and more stable training), we need all the images in the batch to be of the same size. That is why we prefer to either resize or crop them.
#
# Here is a function (well, actually a class), that will apply a transformation 'random crop'. Notice that we apply it to images and masks simultaneously to make sure they correspond, despite the randomness.
#
# Why do we bother making a bulky class to handle the relatively simple task of loading images? We want to keep the code modular. We want to write one dataset object, and then we can try all the possible transforms with this one dataset. Similarly, we want to write one Randomcrop transform object, and then we can reuse it for any other image datasets we night have in the future.
#

# %% [markdown]
# PS: PyTorch already has quite a bunch of all possible data transforms, so if you need one, check [here](https://pytorch.org/vision/stable/transforms.html#transforms-on-pil-image-and-torch-tensor). The biggest problem with them is that they are clearly separated into transforms applied to PIL images (remember, we initially load the images as PIL.Image?) and torch.tensors (remember, we converted the images into tensors by calling transforms.ToTensor()?). This can be incredibly annoying if for some reason you might need to transorm your images to tensors before applying any other transforms or you don't want to use PIL library at all.

# %%
train_data = NucleiDataset(TRAIN_DATA_PATH, transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

# %%
show_random_dataset_image(train_data)

# %% [markdown]
# And the same for the validation data:

# %%
VAL_DATA_PATH = "nuclei_val_data"
val_data = NucleiDataset(VAL_DATA_PATH, transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5)


# %%
show_random_dataset_image(val_data)

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 1 </h2>
# <p>We will go over the steps up to this point soon. If you have time to spare, consider experimenting
# with various augmentations. Your goal is to augment your training data in such a way that you expand
# the distribution of training data to cover the distribution of the rest of your data and avoid overly
# relying on extrapolation at test time. If this sounds somewhat vague thats because augmenting is a
# bit of an artform. Common augmentations that have been shown to work well are adding noise, mirror,
# transpose, and rotations.</p>
# <p>Note that some of these transformations need to be applied to both the raw image and the segmentation,
# wheras others should only be applied to the image (i.e. noise augmentation). The Dataset <code>__init__</code>
# function takes 2 arguments, <code>transform</code> and <code>img_transform</code> so that you can define
# a set of transformations that you only want to apply to the image.</p>
#
# </div>

# %%
augmented_data = NucleiDataset(
    TRAIN_DATA_PATH,
    transforms.RandomCrop(256),
    img_transform=transforms.Compose([transforms.GaussianBlur(5)]),
)


# %%
show_random_dataset_image(augmented_data)

# %% [markdown]
# <hr style="height:2px;">

# %% [markdown]
# ## The model: U-Net
#
# Now we need to define the architecture of the model to use. We will use a [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) that has proven to steadily outperform the other architectures in segmenting biological and medical images.
#
# The image of the model precisely describes all the building blocks you need to use to create it. All of them can be found in the list of PyTorch layers (modules) [here](https://pytorch.org/docs/stable/nn.html#convolution-layers).
#
# The U-Net has an encoder-decoder structure:
#
# In the encoder pass, the input image is successively downsampled via max-pooling. In the decoder pass it is upsampled again via transposed convolutions.
#
# In addition, it has skip connections, that bridge the output from an encoder to the corresponding decoder.


# %%
class UNet(nn.Module):
    """U-Net implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # _conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply two 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1, depth=4, final_activation=None):
        super().__init__()

        assert depth < 10, "Max supported depth is 9"

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = depth

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(
                final_activation, nn.Module
            ), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList(
            [
                self._conv_block(in_channels, 16),
                self._conv_block(16, 32),
                self._conv_block(32, 64),
                self._conv_block(64, 128),
                self._conv_block(128, 256),
                self._conv_block(256, 512),
                self._conv_block(512, 1024),
                self._conv_block(1024, 2048),
                self._conv_block(2048, 4096),
            ][:depth]
        )
        # the base convolution block
        if depth >= 1:
            self.base = self._conv_block(2 ** (depth + 3), 2 ** (depth + 4))
        else:
            self.base = self._conv_block(1, 2 ** (depth + 4))
        # modules of the decoder path
        self.decoder = nn.ModuleList(
            [
                self._conv_block(8192, 4096),
                self._conv_block(4096, 2048),
                self._conv_block(2048, 1024),
                self._conv_block(1024, 512),
                self._conv_block(512, 256),
                self._conv_block(256, 128),
                self._conv_block(128, 64),
                self._conv_block(64, 32),
                self._conv_block(32, 16),
            ][-depth:]
        )

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList(
            [
                self._upsampler(8192, 4096),
                self._upsampler(4096, 2048),
                self._upsampler(2048, 1024),
                self._upsampler(1024, 512),
                self._upsampler(512, 256),
                self._upsampler(256, 128),
                self._upsampler(128, 64),
                self._upsampler(64, 32),
                self._upsampler(32, 16),
            ][-depth:]
        )
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv2d(16, out_channels, 1)
        self.activation = final_activation

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.1</b>: Spot the best U-Net
#
# In the next cell you fill find a series of U-Net definitions. Most of them won't work. Some of them will work but not well. One will do well. Can you identify which model is the winner? Unfortunately you can't yet test your hypotheses yet since we have not covered loss functions, optimizers, and train/validation loops.
#
# </div>

# %%
unetA = UNet(
    in_channels=1, out_channels=1, depth=4, final_activation=torch.nn.Sigmoid()
)
unetB = UNet(in_channels=1, out_channels=1, depth=9, final_activation=None)
unetC = torch.nn.Sequential(
    UNet(in_channels=1, out_channels=1, depth=4, final_activation=torch.nn.ReLU()),
    torch.nn.Sigmoid(),
)
unetD = torch.nn.Sequential(
    UNet(in_channels=1, out_channels=1, depth=1, final_activation=None),
    torch.nn.Sigmoid(),
)


# %%
# Provide your guesses as to what, if anything, might go wrong with each of these models:
#
# unetA: 
#
# unetB: 
#
# unetC: 
#
# unetD: 

favorite_unet: UNet = ...

# %% tags=["solution"]
# Provide your guesses as to what, if anything, might go wrong with each of these models:
#
# unetA: The correct unet.
#
# unetB: Too deep. You won't be able to train with input size 256 since the lowest level will get zero sized tensors.
#
# unetC: A classic mistake putting a Sigmoid after a Relu activation. You will never predict anything < 0.5
#
# unetD: barely any depth to this unet. It should train and give you what you want, I just wouldn't expect good performance

favorite_unet: UNet = unetA

# %% [markdown]
# <hr style="height:2px;">

# %% [markdown]
# ## Loss Function
#
# The next step to do would be writing a loss function - a metric that will tell us how close we are to the desired output. This metric should be differentiable, since this is the value to be backpropagated. The are [multiple losses](https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html) we could use for the segmentation task.
#
# Take a moment to think which one is better to use. If you are not sure, don't forget that you can always google! Before you start implementing the loss yourself, take a look at the [losses](https://pytorch.org/docs/stable/nn.html#loss-functions) already implemented in PyTorch. You can also look for implementations on GitHub.

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.2</b>: Implement your loss (or take one from pytorch):
# </div>

# %%
# implement your loss here or initialize the one of your choice from PyTorch
loss_function: torch.nn.Module = ...

# %% tags=["solution"]
# implement your loss here or initialize the one of your choice from PyTorch
loss_function: torch.nn.Module = nn.BCELoss()

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Test your loss function here, is it behaving as you'd expect?
# </div>

# %%
target = torch.tensor([0.0, 1.0])
good_prediction = torch.tensor([0.01, 0.99])
bad_prediction = torch.tensor([0.4, 0.6])
wrong_prediction = torch.tensor([0.9, 0.1])

good_loss = loss_function(good_prediction, target)
bad_loss = loss_function(bad_prediction, target)
wrong_loss = loss_function(wrong_prediction, target)

assert good_loss < bad_loss
assert bad_loss < wrong_loss

# Can your loss function handle predictions outside of (0, 1)?
# Some loss functions will be perfectly happy with this which may
# make them easier to work with, but predictions outside the expected
# range will not work well with our soon to be discussed evaluation metric.
out_of_bounds_prediction = torch.tensor([-0.1, 1.1])

try:
    oob_loss = loss_function(out_of_bounds_prediction, target)
    print("Your loss supports out-of-bounds predictions.")
except RuntimeError as e:
    print(e)
    print("Your loss does not support out-of-bounds predictions")

# %% [markdown]
# We will use the [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) to evaluate the network predictions.
# We can use it for validation if we interpret set $a$ as predictions and $b$ as labels. It is often used to evaluate segmentations with sparse foreground, because the denominator normalizes by the number of foreground pixels.
# The Dice Coefficient is closely related to Jaccard Index / Intersection over Union.


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.3</b>: Fill in implementation details for the Dice Coefficient
# </div>


# %%
# Sorensen Dice Coefficient implemented in torch
# the coefficient takes values in two discrete arrays
# with values in {0, 1}, and produces a score in [0, 1]
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = ...
        union = ...
        return 2 * intersection / union.clamp(min=self.eps)


# %% tags=["solution"]
# sorensen dice coefficient implemented in torch
# the coefficient takes values in two discrete arrays
# with values in {0, 1}, and produces a score in [0, 1]
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        union = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / union.clamp(min=self.eps)


# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Test your Dice Loss here, are you getting the right scores?
# </div>

# %%
dice = DiceCoefficient()
target = torch.tensor([0.0, 1.0])
good_prediction = torch.tensor([0.0, 1.0])
bad_prediction = torch.tensor([0.0, 0.0])
wrong_prediction = torch.tensor([1.0, 0.0])

assert dice(good_prediction, target) == 1.0, dice(good_prediction, target)
assert dice(bad_prediction, target) == 0.0, dice(bad_prediction, target)
assert dice(wrong_prediction, target) == 0.0, dice(wrong_prediction, target)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.4</b>: What happes if your predictions are not discrete elements of {0,1}?
#     <ol>
#         <li>What if the predictions are in range (0,1)?</li>
#         <li>What if the predictions are in range ($-\infty$,$\infty$)?</li>
#     </ol>
# </div>

# %% [markdown]
# Answer:
# 1) ...
#
# 2) ...

# %% [markdown] tags=["solution"]
# Answer:
# 1) Score remains between (0,1) with 0 being the worst score and 1 being the best. This case essentially gives you the Dice Loss and can be a good alternative to cross entropy.
#
# 2) Scores will fall in the range of [-1,1]. Overly confident scores will be penalized i.e. if the target is `[0,1]` then a prediction of `[0,2]` will score lower than a prediction of `[0,3]`.

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 2</h2>
#
# This is a good place to stop for a moment. If you have extra time look into some extra loss functions or try to implement your own if you haven't yet. You could also explore alternatives for evaluation metrics since there are alternatives that you could look into.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# ## Training
#
# Let's start with writing training and validation functions.

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 3.1</b>: Fix in all the TODOs to make the train function work. If confused, you can use this [PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) as a template
# </div>


# %%
# apply training for one epoch
def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # apply model and calculate loss
        prediction = ...  # placeholder since we use prediction later
        loss = ...  # placeholder since we use the loss later

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


# %% tags=["solution"]
# apply training for one epoch
def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Quick sanity check for your train function to make sure no errors are thrown:
#     Good place to test unetA, unetB, unetC, unetD to see if you can eliminate some
# </div>


# %%
simple_net = UNet(1, 1, depth=1, final_activation=nn.Sigmoid())

train(
    simple_net,
    train_loader,
    optimizer=torch.optim.Adam(simple_net.parameters()),
    loss_function=torch.nn.MSELoss(),
    epoch=0,
    log_interval=1,
    early_stop=True,
)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 3.2</b>: Fix in all the TODOs to make the validate function work. If confused, you can use this <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html">PyTorch tutorial</a> as a template
# </div>


# %%
# run validation after training epoch
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # TODO: evaluate this example with the given loss and metric
            prediction = ...
            val_loss += ...
            val_metric += ...

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )


# %% tags=["solution"]
# run validation after training epoch
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y).item()
            val_metric += metric(prediction > 0.5, y).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )


# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Quick sanity check for your train function to make sure no errors are thrown
# </div>

# %%
simple_net = UNet(1, 1, depth=1, final_activation=None)

# build the dice coefficient metric

validate(
    simple_net,
    train_loader,
    loss_function=torch.nn.MSELoss(),
    metric=DiceCoefficient(),
    step=0,
)

# %% [markdown]
#
# We want to use GPU to train faster. Make sure GPU is available

# %%
assert torch.cuda.is_available()

# %%
# start a tensorboard writer
logger = SummaryWriter('runs/Unet')
%tensorboard --logdir runs

# %%
# Use the unet you expect to work the best!
model = favorite_unet

# use adam optimizer
optimizer = torch.optim.Adam(model.parameters())

# build the dice coefficient metric
metric = DiceCoefficient()

# train for $25$ epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 25
for epoch in range(n_epochs):
    # train
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    # validate
    validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)


# %% [markdown]
# Your validation metric was probably around 85% by the end of the training. That sounds good enough, but an equally important thing to check is:
# Open the Images tab in your Tensorboard and compare predictions to targets. Do your predictions look reasonable? Are there any obvious failure cases?
# If nothing is clearly wrong, let's see if we can still improve the model performance by changing the model or the loss
#

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 3</h2>
#
# This is the end of the guided exercise. We will go over all of the code up until this point shortly. While you wait you are encouraged to try alternative loss functions, evaluation metrics, augmentations, and networks.
# After this come additional exercises if you are interested and have the time.
#
# </div>
# <hr style="height:2px;">

# %% [markdown]
# ## Additional Exercises
#
# 1. Modify and evaluate the following architecture variants of the U-Net:
#     * use [GroupNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.GroupNorm) to normalize convolutional group inputs
#     * use more layers in your U-Net.
#
# 2. Use the Dice Coefficient as loss function. Before we only used it for validation, but it is differentiable and can thus also be used as loss. Compare to the results from exercise 2.
# Hint: The optimizer we use finds minima of the loss, but the minimal value for the Dice coefficient corresponds to a bad segmentation. How do we need to change the Dice Coefficient to use it as loss nonetheless?
#
# 3. Compare the results of these trainings to the first one. If any of the modifications you've implemented show better results, combine them (e.g. add both GroupNorm and one more layer) and run trainings again.
# What is the best result you could get?

# %% [markdown]

# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.1</b>: Group Norm, update the U-Net to use a GroupNorm layer
# </div>


# %%
class UNetGN(UNet):
    """
    A subclass of UNet that implements GroupNorm in each convolutional block
    """

    # Convolutional block for single layer of the decoder / encoder
    # we apply two 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        # See the original U-Net for an example of how to build the convolutional block
        # We want operation -> activation -> normalization (2x)
        # Hint: Group norm takes a "num_groups" argument. Use 8 to match the solution
        return ...


# %% tags=["solution"]
class UNetGN(UNet):
    """
    A subclass of UNet that implements GroupNorm in each convolutional block
    """

    # Convolutional block for single layer of the decoder / encoder
    # we apply two 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels),
        )


# %%
model = UNetGN(1, 1, final_activation=nn.Sigmoid())

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

logger = SummaryWriter("runs/UNetGN")


# train for 40 epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 40
for epoch in range(n_epochs):
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.2</b>: More Layers
# </div>

# %%
# Experiment with more layers. For example UNet with depth 5

model = ...

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

loss = torch.nn.BCELoss()

logger = SummaryWriter("runs/UNet5layers")

# %% tags=["solution"]
# Experiment with more layers. For example UNet with depth 5

model = UNet(1, 1, depth=5, final_activation=nn.Sigmoid())

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

loss = torch.nn.BCELoss()

logger = SummaryWriter("runs/UNet5layers")

# %%
# train for 25 epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 25
for epoch in range(n_epochs):
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(model, val_loader, loss, metric, step=step, tb_logger=logger)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.3</b>: Dice Loss
#     Dice Loss is a simple inversion of the Dice Coefficient.
#     We already have a Dice Coefficient implementation, so now we just
#     need a layer that can invert it.
# </div>


# %%
class DiceLoss(nn.Module):
    """ """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y):
        ...


# %% tags=["solution"]
class DiceLoss(nn.Module):
    """
    This layer will simply compute the dice coefficient and then negate
    it with an optional offset.
    We support an optional offset because it is common to have 0 as
    the optimal loss. Since the optimal dice coefficient is 1, it is
    convenient to get 1 - dice_coefficient as our loss.

    You could leave off the offset and simply have -1 as your optimal loss.
    """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(offset), requires_grad=False)
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y):
        coefficient = self.dice_coefficient(x, y)
        return self.offset - coefficient


# %%
# Now combine the Dice Coefficient layer with the Invert layer to make a Dice Loss
dice_loss = ...


# %% tags=["solution"]
# Now combine the Dice Coefficient layer with the Invert layer to make a Dice Loss
dice_loss = DiceLoss()

# %%
# Experiment with Dice Loss
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
# Experiment with Dice Loss
net = UNet(1, 1, final_activation=nn.Sigmoid())
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

# %%
logger = SummaryWriter("runs/UNet_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(net, val_loader, loss_func, metric, step=step, tb_logger=logger)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.4</b>: Group Norm + Dice
# </div>

# %%
net = ...
optimizer = ...
metric = ...
loss_func = ...

logger = SummaryWriter("runs/UNetGN_diceloss")

# %% tags=["solution"]
net = UNetGN(1, 1, final_activation=nn.Sigmoid())
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

# %%
logger = SummaryWriter("runs/UNetGN_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(net, val_loader, loss_func, metric, step=step, tb_logger=logger)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.5</b>: Group Norm + Dice + U-Net 5 Layers
# </div>

# %%
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
net = UNetGN(1, 1, depth=5, final_activation=nn.Sigmoid())
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

logger = SummaryWriter("runs/UNet5layersGN_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(net, val_loader, loss_func, metric, step=step, tb_logger=logger)
