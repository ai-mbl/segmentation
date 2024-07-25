# %% [markdown]
# # Semantic Segmentation
#
# <hr style="height:2px;">
#
# In this notebook, we adapt our 2D U-Net for better nuclei segmentations in the Kaggle Nuclei dataset.
#
# <div class="alert alert-info">
#
# **Specifically, you will:**
#
# 1. Prepare the 2D U-Net baseline model and validation dataset.
# 2. Implement and use the Dice coefficient as an evaluation metric for the baseline model.
# 3. Improve metrics by experimenting with:
#     - Data augmentations
#     - Loss functions
#     - (bonus) Group Normalization, U-Net architecture
# </div>
# Written by William Patton, Valentyna Zinchenko, and Constantin Pape.

# %% [markdown]
# Our goal is to produce a model that can take an image as input and produce a segmentation as shown in this table.
#
# | Image | Mask | Prediction |
# | :-: | :-: | :-: |
# | ![image](static/img_0.png) | ![mask](static/mask_0.png) | ![pred](static/pred_0.png) |
# | ![image](static/img_1.png) | ![mask](static/mask_1.png) | ![pred](static/pred_1.png) |

# %% [markdown]
# <hr style="height:2px;">
#
# ## The libraries

# %%
# %matplotlib inline
# %load_ext tensorboard
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import torchvision.transforms.v2 as transforms_v2

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()


# %% [markdown]
# ## Section 0: What we have so far
# You have already implemented a U-Net architecture in the previous exercise. We will use it as a starting point for this exercise.
# You should also alredy have the dataset and the dataloader implemented, along with a simple train loop with MSELoss.
# Lets go ahead and visualize some of the data along with some predictions to see how we are doing.


# %%
from local import (
    NucleiDataset,
    show_random_dataset_image,
    show_random_dataset_image_with_prediction,
    show_random_augmentation_comparison,
    train,
)
from unet import UNet


# %% [markdown]
#
# *Note*: We are artificially making our validation data worse. This dataset
# was chosen to be reasonable to segment in the amount of time it takes to
# run this exercise. However this means that some techniques like augmentations
# aren't as useful as they would be on a more complex dataset. So we are
# artificially adding noise to the validation data to make it more challenging.


# %%
def salt_and_pepper_noise(image, amount=0.05):
    """
    Add salt and pepper noise to an image
    """
    out = image.clone()
    num_salt = int(amount * image.numel() * 0.5)
    num_pepper = int(amount * image.numel() * 0.5)

    # Add Salt noise
    coords = [
        torch.randint(0, i - 1, [num_salt]) if i > 1 else [0] * num_salt
        for i in image.shape
    ]
    out[coords] = 1

    # Add Pepper noise
    coords = [
        torch.randint(0, i - 1, [num_pepper]) if i > 1 else [0] * num_pepper
        for i in image.shape
    ]
    out[coords] = 0

    return out


# %%
train_data = NucleiDataset("nuclei_train_data", transforms_v2.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)
val_data = NucleiDataset(
    "nuclei_val_data",
    transforms_v2.RandomCrop(256),
    img_transform=transforms_v2.Lambda(salt_and_pepper_noise),
)
val_loader = DataLoader(val_data, batch_size=5)

# %%
unet = UNet(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters())

for epoch in range(10):
    train(unet, train_loader, optimizer, loss, epoch, device=device)


# %%
# Show some predictions on the train data
show_random_dataset_image(train_data)
show_random_dataset_image_with_prediction(train_data, unet, device)
# %%
# Show some predictions on the validation data
show_random_dataset_image(val_data)
show_random_dataset_image_with_prediction(val_data, unet, device)

# %% [markdown]
#
# <div class="alert alert-block alert-info">
#     <p><b>Task 0.1</b>: Are the predictions good enough? Take some time to try to think about
#     what could be improved and how that could be addressed. If you have time try training a second
#     model and see which one is better</p>
# </div>


# %% [markdown]
# Write your answers here:
# <ol>
#     <li></li>
#     <li></li>
#     <li></li>
# </ol>

# %% [markdown] tags=["solution"]
# Write your answers here:
# <ol>
#     <li> Evaluation metric for better understanding of model performance so we can compare. </li>
#     <li> Augments for generalization to validaiton. </li>
#     <li> Loss function for better performance on lower prevalence classes. </li>
# </ol>

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 0 </h2>
# <p>We will go over the steps up to this point soon. By this point you should have imported and re-used
# code from previous exercises to train a basic UNet.</p>
# <p>The rest of this exercise will focus on tailoring our network to semantic segmentation to improve
# performance. The main areas we will tackle are:</p>
# <ol>
#   <li> Evaluation
#   <li> Augmentation
#   <li> Activations/Loss Functions
# </ol>
#
# </div>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 1: Evaluation

# %% [markdown]
# One of the most important parts of training a model is evaluating it. We need to know how well our model is doing and if it is improving.
# We will start by implementing a metric to evaluate our model. Evaluation is always specific to the task, in this case semantic segmentation.
# We will use the [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) to evaluate the network predictions.
# We can use it for validation if we interpret set $a$ as predictions and $b$ as labels. It is often used to evaluate segmentations with sparse
# foreground, because the denominator normalizes by the number of foreground pixels.
# The Dice Coefficient is closely related to Jaccard Index / Intersection over Union.
# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.1</b>: Fill in implementation details for the Dice Coefficient
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

    # the dice coefficient of two sets represented as vectors a, b can be
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
#     Test your Dice Coefficient here, are you getting the right scores?
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
# <b>Task 1.2</b>: What happens if your predictions are not discrete elements of {0,1}?
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
# 1) Score remains between (0,1) with 0 being the worst score and 1 being the best. This case
# essentially gives you the Dice Loss and can be a good alternative to cross entropy.
#
# 2) Scores will fall in the range of [-1,1]. Overly confident scores will be penalized i.e.
# if the target is `[0,1]` then a prediction of `[0,2]` will score higher than a prediction of `[0,3]`.

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 1.1 </h2>
#
# This is a good place to stop for a moment. If you have extra time look into some extra
# evaluation functions or try to implement your own without hints.
# Some popular alternatives to the Dice Coefficient are the Jaccard Index and Balanced F1 Scores.
# You may even have time to compute the evaluation score between some of your training and
# validation predictions to their ground truth using our previous models.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 1.3</b>: Fix in all the TODOs to make the validate function work. If confused, you can use this
# <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html">PyTorch tutorial</a> as a template
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
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
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
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
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
# <div class="alert alert-block alert-info">
#     <b>Task 1.4</b>: Evaluate your first model using the Dice Coefficient. How does it perform? If you trained two models,
#     do the scores agree with your visual determination of which model was better?
# </div>

# %%

# Evaluate your model here

# %% tags=["solution"]

# Evaluate your model here

validate(
    unet,
    val_loader,
    loss_function=torch.nn.MSELoss(),
    metric=DiceCoefficient(),
    step=0,
    device=device,
)

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 1.2</h2>
#
# We have finished writing the evaluation function. We will go over the code up to this point soon.
# Next we will work on augmentations to improve the generalization of our model.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# ## Section 2: Augmentation
# Often our models will perform better on the evaluation dataset if we augment our training data.
# This is because the model will be exposed to a wider variety of data that will hopefully help
# cover the full distribution of data in the validation set. We will use the `torchvision.transforms`
# to augment our data.


# %% [markdown]
# PS: PyTorch already has quite a few possible data transforms, so if you need one, check
# [here](https://pytorch.org/vision/stable/transforms.html#transforms-on-pil-image-and-torch-tensor).
# The biggest problem with them is that they are clearly separated into transforms applied to PIL
# images (remember, we initially load the images as PIL.Image?) and torch.tensors (remember, we
# converted the images into tensors by calling transforms.ToTensor()?). This can be incredibly
# annoying if for some reason you might need to transorm your images to tensors before applying any
# other transforms or you don't want to use PIL library at all.

# %% [markdown]
# Here is an example augmented dataset. Use it to see how it affects your data, then play around with at least
# 2 other augmentations.
# There are two types of augmentations: `transform` and `img_transform`. The first one is applied to both the
# image and the mask, the second is only applied to the image. This is useful if you want to apply augmentations
# that spatially distort your data and you want to make sure the same distortion is applied to the mask and image.
# `img_transform` is useful for augmentations that don't make sense to apply to the mask, like blurring.

# %%
train_data = NucleiDataset("nuclei_train_data", transforms_v2.RandomCrop(256))

# Note this augmented data uses extreme augmentations for visualization. It will not train well
example_augmented_data = NucleiDataset(
    "nuclei_train_data",
    transforms_v2.Compose(
        [transforms_v2.RandomRotation(45), transforms_v2.RandomCrop(256)]
    ),
    img_transform=transforms_v2.Compose([transforms_v2.GaussianBlur(21, sigma=10.0)]),
)

# %%
show_random_augmentation_comparison(train_data, example_augmented_data)

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.1</b>: Now create an augmented dataset with an augmentation of your choice.
#       **hint**: Using the same augmentation as was applied to the validation data will
#      likely be optimal. Bonus points if you can get good results without the custom noise.
# </div>

# %%
augmented_data = ...

# %% tags=["solution"]
augmented_data = NucleiDataset(
    "nuclei_train_data",
    transforms_v2.Compose(
        [transforms_v2.RandomRotation(45), transforms_v2.RandomCrop(256)]
    ),
    img_transform=transforms_v2.Compose([transforms_v2.Lambda(salt_and_pepper_noise)]),
)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.2</b>: Now retrain your model with your favorite augmented dataset. Did your model improve?
# </div>

# %%

unet = UNet(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters())
augmented_loader = DataLoader(augmented_data, batch_size=5, shuffle=True, num_workers=8)

...

# %% tags=["solution"]

unet = UNet(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters())
augmented_loader = DataLoader(augmented_data, batch_size=5, shuffle=True, num_workers=8)

for epoch in range(10):
    train(unet, augmented_loader, optimizer, loss, epoch, device=device)

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.3</b>: Now evaluate your model. Did your model improve?
# </div>

# %%
validate(...)

# %% tags=["solution"]
validate(unet, val_loader, loss, DiceCoefficient(), device=device)

# %% [markdown]
# <hr style="height:2px;">

# %% [markdown]
# ## Section 3: Loss Functions

# %% [markdown]
# The next step to do would be to improve our loss function - the metric that tells us how
# close we are to the desired output. This metric should be differentiable, since this
# is the value to be backpropagated. The are
# [multiple losses](https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html)
# we could use for the segmentation task.
#
# Take a moment to think which one is better to use. If you are not sure, don't forget
# that you can always google! Before you start implementing the loss yourself, take a look
# at the [losses](https://pytorch.org/docs/stable/nn.html#loss-functions) already implemented
# in PyTorch. You can also look for implementations on GitHub.

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 3.1</b>: Implement your loss (or take one from pytorch):
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
# Pay close attention to whether your loss function can handle predictions outside of the range (0, 1).
# If it can't, theres a good chance that the activation function requires a specific activation before
# being passed into the loss function. This is a common source of bugs in DL models. For example, trying
# to use the `torch.nn.BCEWithLogitsLoss` loss function with a model that has a sigmoid activation will
# result in abysmal performance, wheras using the `torch.nn.BCELoss` loss function with a model that has
# no activation function will likely error out and fail to train.


# %%
# Now lets start experimenting. Start a tensorboard logger to keep track of experiments.
# start a tensorboard writer
logger = SummaryWriter("runs/Unet")
# %tensorboard --logdir runs


# %%
# Use the unet you expect to work the best!
model = UNet(
    depth=4, in_channels=1, out_channels=1, num_fmaps=2, final_activation="Sigmoid"
).to(device)

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
        log_interval=25,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    # validate
    validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)


# %% [markdown]
# Your validation metric was probably around 85% by the end of the training. That sounds good enough,
# but an equally important thing to check is: Open the Images tab in your Tensorboard and compare
# predictions to targets. Do your predictions look reasonable? Are there any obvious failure cases?
# If nothing is clearly wrong, let's see if we can still improve the model performance by changing
# the model or the loss
#

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 3</h2>
#
# This is the end of the guided exercise. We will go over all of the code up until this point shortly.
# While you wait you are encouraged to try alternative loss functions, evaluation metrics, augmentations,
# and networks. After this come additional exercises if you are interested and have the time.
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
# 2. Use the Dice Coefficient as loss function. Before we only used it for validation, but it is differentiable
# and can thus also be used as loss. Compare to the results from exercise 2.
# Hint: The optimizer we use finds minima of the loss, but the minimal value for the Dice coefficient corresponds
# to a bad segmentation. How do we need to change the Dice Coefficient to use it as loss nonetheless?
#
# 3. Compare the results of these trainings to the first one. If any of the modifications you've implemented show
# better results, combine them (e.g. add both GroupNorm and one more layer) and run trainings again.
# What is the best result you could get?

# %% [markdown]
#
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.1</b>: Group Norm, update the U-Net to use a GroupNorm layer
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
model = UNetGN(
    depth=4, in_channels=1, out_channels=1, num_fmaps=2, final_activation="Sigmoid"
).to(device)

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
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        model,
        val_loader,
        loss_function,
        metric,
        step=step,
        tb_logger=logger,
        device=device,
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.2</b>: More Layers
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

model = UNet(
    depth=5, in_channels=1, out_channels=1, num_fmaps=2, final_activation="Sigmoid"
).to(device)

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
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        model, val_loader, loss, metric, step=step, tb_logger=logger, device=device
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.3</b>: Dice Loss
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

    def forward(self, x, y): ...


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
net = UNet(
    depth=4, in_channels=1, out_channels=1, num_fmaps=2, final_activation="Sigmoid"
).to(device)
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
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        net, val_loader, loss_func, metric, step=step, tb_logger=logger, device=device
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.4</b>: Group Norm + Dice
# </div>

# %%
net = ...
optimizer = ...
metric = ...
loss_func = ...

logger = SummaryWriter("runs/UNetGN_diceloss")

# %% tags=["solution"]
net = UNetGN(
    depth=4, in_channels=1, out_channels=1, num_fmaps=2, final_activation="Sigmoid"
).to(device)
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
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        net, val_loader, loss_func, metric, step=step, tb_logger=logger, device=device
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.5</b>: Group Norm + Dice + U-Net 5 Layers
# </div>

# %%
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
net = UNetGN(
    depth=5, in_channels=1, out_channels=1, num_fmaps=2, final_activation="Sigmoid"
).to(device)
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
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        net, val_loader, loss_func, metric, step=step, tb_logger=logger, device=device
    )
