import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore 
from tensorflow.keras.layers import Flatten, BatchNormalization # type: ignore 
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Conv2D # type: ignore 
from tensorflow.keras.layers import LeakyReLU # type: ignore 
from tensorflow.keras.layers import Activation # type: ignore 
from tensorflow.keras.layers import Concatenate # type: ignore 
from tensorflow.keras.layers import UpSampling2D, Conv2D # type: ignore 
from tensorflow.keras.layers import Activation, ZeroPadding2D # type: ignore 

def build_discriminator(image_shape=(160, 160, 3)):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

    return model
