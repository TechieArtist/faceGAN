import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def visualize_feature_maps_by_index(model, layer_index, input_image, num_columns=16):
    """
    Visualizes the feature maps of a specified layer by index for a given input image.

    :param model: The model from which to visualize the feature maps.
    :param layer_index: The index of the layer to visualize.
    :param input_image: The input image for which feature maps are visualized.
    :param num_columns: Number of columns in the subplot grid.
    """
    # Ensure the layer index is within the valid range
    if layer_index < 0 or layer_index >= len(model.layers):
        print(f"Layer index {layer_index} is out of bounds for the model.")
        return

    # Extract the outputs of the specified layer
    layer_output = model.layers[layer_index].output

    # Create a model for feature map extraction
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)

    # Preprocess input image and predict
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension if needed
    activations = activation_model.predict(input_image)

    # Number of feature maps to visualize
    num_feature_maps = activations.shape[-1]

    # Calculate the number of rows
    num_rows = num_feature_maps // num_columns + (num_feature_maps % num_columns > 0)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns, num_rows))
    fig.suptitle('Feature Maps')

    # Plot each feature map
    for i in range(num_feature_maps):
        row = i // num_columns
        col = i % num_columns
        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.imshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
