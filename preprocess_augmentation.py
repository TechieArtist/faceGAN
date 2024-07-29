from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image

def augment_and_save_data(original_data_path, augmented_data_path, image_size, num_augmentations):
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Ensure the directory for augmented data exists
    os.makedirs(augmented_data_path, exist_ok=True)

    # Load and augment each image
    for filename in os.listdir(original_data_path):
        img_path = os.path.join(original_data_path, filename)
        img = Image.open(img_path).resize(image_size)
        img_array = np.array(img).reshape((1,) + img.size + (3,))  # Assuming color images

        for _ in range(num_augmentations):
            # This will loop 'num_augmentations' times for each image
            for batch in data_gen.flow(img_array, batch_size=1,
                                       save_to_dir=augmented_data_path,
                                       save_prefix='aug_' + filename,
                                       save_format='jpeg'):
                break  # We only need to save one image per augmentation

# Example usage
original_data_path = '/content/drive/MyDrive/archive/images/'
augmented_data_path = '/content/drive/MyDrive/augmented_archive/'
augment_and_save_data(original_data_path, augmented_data_path, (160, 160), 5)
