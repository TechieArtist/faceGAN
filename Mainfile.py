import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense # type: ignore
from tensorflow.keras.layers import Flatten, BatchNormalization # type: ignore 
from tensorflow.keras.layers import Activation, ZeroPadding2D # type: ignore 
from tensorflow.keras.layers import LeakyReLU # type: ignore 
from tensorflow.keras.layers import UpSampling2D, Conv2D # type: ignore 
from tensorflow.keras.models import Sequential, Model, load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore 
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

# Generation resolution - Must be square
GENERATE_RES = 5  # Generation resolution factor (1=32, 2=64, 3=96, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES  # rows/cols (should be square)
IMAGE_CHANNELS = 3

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100


# Configuration
DATA_PATH = './data'  # Path to your local dataset
EPOCHS = 250
CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PERIOD = 70
BATCH_SIZE = 64
BUFFER_SIZE = 60000

print(f"Will generate {GENERATE_SQUARE}px square images.")

# Directory where augmented images are saved
augmented_data_dir = './augmented_data'

# Load Images Function
def load_images_from_directory(directory, image_size):
    images = []
    for filename in tqdm(os.listdir(directory)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).resize(image_size)
            img_array = np.asarray(img)
            images.append(img_array)
    return np.array(images)

# Assuming you have resized images to a uniform size
image_size = (160, 160)

# Load original and augmented images
original_images = load_images_from_directory(os.path.join(DATA_PATH, 'images'), image_size)
augmented_images = load_images_from_directory(augmented_data_dir, image_size)

print("All images loaded.")
print(f"Total number of images loaded: {len(augmented_images)}")

# Combine and normalize datasets
combined_images = np.concatenate((original_images, augmented_images), axis=0)
combined_images = combined_images.astype(np.float32) / 127.5 - 1

# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(combined_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# After loading and processing augmented_images
print("Augmented data shape:", np.array(augmented_images).shape)

# Initialize the GAN components with error handling
try:
    generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)
    discriminator = build_discriminator((GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
except Exception as e:
    print(f"Error in model building: {e}")
    raise

from visualization_utils import visualize_feature_maps_by_index

# Optimizers
generator_optimizer = tf.keras.optimizers.AdamW(learning_rate=1.7e-4, weight_decay=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-07, name='AdamW')
discriminator_optimizer = tf.keras.optimizers.AdamW(learning_rate=1.5e-4, weight_decay=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-07, name='AdamW')

# Initialize the GAN components
generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)
discriminator = build_discriminator((GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

# Add an epoch counter variable
current_epoch = tf.Variable(0, dtype=tf.int64)

# Initialize checkpoints
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 current_epoch=current_epoch)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

# Attempt to restore the latest checkpoint
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    start_epoch = current_epoch.numpy()
    print(f"Restored from {checkpoint_manager.latest_checkpoint} at epoch {start_epoch}")
else:
    print("Initializing from scratch.")
    start_epoch = 0

# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Load InceptionV3 model
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

def calculate_fid(model, real_images, generated_images):
    # Scale images
    real_images = (real_images * 127.5 + 127.5) / 255.0
    generated_images = (generated_images * 127.5 + 127.5) / 255.0

    # Resize images
    real_images = np.array([resize(image, (299, 299, 3)) for image in real_images])
    generated_images = np.array([resize(image, (299, 299, 3)) for image in generated_images])

    # Preprocess images
    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)

    # Calculate activations
    act1 = model.predict(real_images)
    act2 = model.predict(generated_images)

    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Compute sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

smooth_factor = 0.9

@tf.function
def train_step(images):
    batch_size = tf.shape(images)[0]  # Dynamically get the batch size

    # Generate smoothed labels with the dynamic batch size
    real_labels = tf.ones((batch_size, 1)) * smooth_factor
    fake_labels = tf.zeros((batch_size, 1))

    seed = tf.random.normal([batch_size, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Update discriminator loss with dynamically sized labels
        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)
        disc_loss = real_loss + fake_loss

        gen_loss = generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE + PREVIEW_MARGIN)), IMAGE_CHANNELS),
        255, dtype=np.uint8)

    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
            c = col * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
            image_array[r:r + GENERATE_SQUARE, c:c + GENERATE_SQUARE] = generated_images[image_count] * 255
            image_count += 1

    output_path = os.path.join('output', f"trained-{cnt}.png")
    if not os.path.exists('output'):
        os.makedirs('output')
    im = Image.fromarray(image_array)
    im.save(output_path)

# Train the GAN
fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

start = time.time()
for epoch in range(start_epoch, EPOCHS):
    current_epoch.assign(epoch)  # Update the epoch counter variable

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch)
        gen_loss_list.append(gen_loss)
        disc_loss_list.append(disc_loss)

    avg_gen_loss = np.mean(gen_loss_list)
    avg_disc_loss = np.mean(disc_loss_list)

    # Save model at checkpoint
    if (epoch + 1) % CHECKPOINT_PERIOD == 0:
        checkpoint_manager.save()

    print(f'Epoch {epoch+1}, Gen Loss: {avg_gen_loss}, Disc Loss: {avg_disc_loss}')

    if (epoch + 1) % CHECKPOINT_PERIOD == 0:
        save_images(epoch + 1, fixed_seed)

    # Evaluate FID every 50 epochs
    if (epoch + 1) % 50 == 0:
        noise = np.random.normal(0, 1, (len(original_images), SEED_SIZE))
        generated_images = generator.predict(noise)
        fid_score = calculate_fid(inception_model, original_images, generated_images)
        print(f"FID score at epoch {epoch + 1}: {fid_score}")

# Save final models
generator.save(os.path.join(CHECKPOINT_DIR, 'generator_final.h5'))
discriminator.save(os.path.join(CHECKPOINT_DIR, 'discriminator_final.h5'))

print(f'Time taken: {hms_string(time.time() - start)}')
