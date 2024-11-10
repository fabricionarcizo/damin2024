#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This script demonstrates how to apply dimensionality reduction techniques to
a real-world dataset."""

# Import necessary libraries.
from typing import Tuple

import os
import numpy as np
import torch

import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from torchvision.models import resnet50


# Paths to your cat and dog images
cat_dir = "lecture11/data/cats_set"
dog_dir = "lecture11/data/dogs_set"


class ImageDataset(Dataset):
    """
    Custom Dataset for loading images.
    
    Attributes:
        image_dir (str): The directory containing the images.
        image_names (list): The list of image names.
        label (int): The label associated with the images.
        transform (callable): The transformation to be applied to the
    """

    def __init__(self, image_dir: str, label: int, transform: Compose=None):
        """
        Constructor method.

        Args:
            image_dir (str): The directory containing the images.
            label (int): The label associated with the images.
            transform (Compose): The transformation to be applied to the images.
        """
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.label = label
        self.transform = transform
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return the image and its label at the given index.

        Args:
            idx (int): The index of the image to be returned.

        Returns:
            Tuple[torch.Tensor, int]: The image and its label.
        """
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


# Preprocess transformations.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and loaders for both classes.
cat_dataset = ImageDataset(cat_dir, label=0, transform=transform)
dog_dataset = ImageDataset(dog_dir, label=1, transform=transform)
combined_dataset = cat_dataset + dog_dataset  # Combine both datasets.
loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet50 model.
model = resnet50(pretrained=True)

# Remove the final classification layer.
model = torch.nn.Sequential(*list(model.children())[:-1])

# Set model to evaluation mode.
model.eval()

# Extract features.
features = []
labels = []
with torch.no_grad():
    for imgs, lbls in loader:
        outputs = model(imgs)
        outputs = outputs.view(outputs.size(0), -1)
        features.append(outputs.cpu().numpy())
        labels.extend(lbls.numpy())

features = np.vstack(features)
labels = np.array(labels)

# Save features and labels to .npy files.
np.save("lecture11/data/features.npy", features)
np.save("lecture11/data/labels.npy", labels)
