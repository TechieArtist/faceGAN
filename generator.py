# generator.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D

def build_generator(seed_size, channels):
    # Build and return the generator model
    model = Sequential()

    # Start with a 5x5 size, which is easier to scale up to 160x160
    model.add(Dense(5*5*256, activation="relu", input_dim=seed_size))
    model.add(Reshape((5, 5, 256)))

    # Upsample to 10x10
    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Upsample to 20x20
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Upsample to 40x40
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Upsample to 80x80
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Upsample to 160x160
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model
