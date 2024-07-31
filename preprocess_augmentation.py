import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os

def augment_and_save_data(original_data_path, augmented_data_path, image_size, num_augmentations):
    # Define the augmentation transformations
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(contrast=0.2, saturation=0.2),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Ensure the directory for augmented data exists
    os.makedirs(augmented_data_path, exist_ok=True)

    # Process each image
    for filename in os.listdir(original_data_path):
        img_path = os.path.join(original_data_path, filename)
        img = Image.open(img_path).resize(image_size)
        
        for i in range(num_augmentations):
            # Apply the transformations
            augmented_img = transform(img)
            
            # Save the augmented image
            augmented_img_pil = transforms.ToPILImage()(augmented_img)
            augmented_img_pil.save(os.path.join(augmented_data_path, f'aug_{i}_{filename}'))


