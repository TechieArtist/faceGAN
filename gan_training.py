import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam 
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.datasets.mnist import load_data  # or your specific dataset
from skimage.transform import resize

def train(train_dataset, epochs):
    fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
    start = time.time()

    fid_scores = []

    # Define the layer index for generator visualization
    generator_layer_index = 5  # Change as needed

    # Generate a noise vector for visualization
    noise_vector_for_visualization = np.random.normal(0, 1, (SEED_SIZE,))

    FID_INTERVAL = 10  # Interval for calculating FID score
    VISUALIZATION_INTERVAL = 500  # Interval for visualizing feature maps

   # Select a subset of combined data for FID calculation
    num_samples = 4000  # Adjust this number as needed
    subset_indices = np.random.choice(len(combined_images), num_samples, replace=False)
    training_subset = combined_images[subset_indices]


    print("Initializing training...")
    for epoch in range(start_epoch,epochs):
        print(f"Starting epoch {epoch+1}")
        epoch_start = time.time()
        gen_loss_list = []
        disc_loss_list = []
        current_epoch.assign(epoch + 1)


        for image_batch in train_dataset:
            t = train_step(image_batch)
            gen_loss_list.append(t[0].numpy())
            disc_loss_list.append(t[1].numpy())



        # Print the mean generator and discriminator loss for the epoch
        print(f'Epoch {epoch + 1}, gen loss={np.mean(gen_loss_list)}, disc loss={np.mean(disc_loss_list)}, {hms_string(time.time() - epoch_start)}')

        # Calculate FID every FID_INTERVAL epochs
        if (epoch + 1) % FID_INTERVAL == 0:
         noise = np.random.normal(0, 1, (num_samples, SEED_SIZE))
         generated_images = generator.predict(noise)
         fid = calculate_fid(inception_model, training_subset, generated_images)
         print(f'Epoch {epoch + 1}, FID: {fid}')
         fid_scores.append(fid)  # Append the FID score to the list

        # Visualize feature maps every VISUALIZATION_INTERVAL epochs
        if (epoch + 1) % VISUALIZATION_INTERVAL == 0:
         visualize_feature_maps_by_index(generator, generator_layer_index, noise_vector_for_visualization)
          # Save the model every CHECKPOINT_PERIOD epochs
        if (epoch + 1) % CHECKPOINT_PERIOD == 0:
            save_path = checkpoint_manager.save()
            print(f"Saved checkpoint for epoch {epoch + 1}: {save_path}")


        # Save the generated images
        save_images(epoch, fixed_seed)

    print(f'Total Training time: {hms_string(time.time() - start)}')

    # Plot the FID scores
    plt.figure(figsize=(10,5))
    plt.title("FID Scores over Epochs")
    plt.plot(fid_scores, label="FID Score")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.legend()
    plt.show()
