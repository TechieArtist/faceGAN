import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def visualize_feature_maps_by_index(model, layer_index, input_image, num_columns=16):
    """
    Visualizes the feature maps of a specified layer by index for a given input image.

    :param model: The PyTorch model from which to visualize the feature maps.
    :param layer_index: The index of the layer to visualize.
    :param input_image: The input image for which feature maps are visualized.
    :param num_columns: Number of columns in the subplot grid.
    """
    # Ensure the layer index is within the valid range
    if layer_index < 0 or layer_index >= len(list(model.children())):
        print(f"Layer index {layer_index} is out of bounds for the model.")
        return

    # Hook to get feature maps
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    # Register hook
    feature_maps = None
    hooks = []
    layers = list(model.children())
    hook = layers[layer_index].register_forward_hook(hook_fn)
    hooks.append(hook)

    # Preprocess input image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])
    input_image = preprocess(input_image).unsqueeze(0)  # Add batch dimension

    # Perform a forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    # Remove the hook
    for hook in hooks:
        hook.remove()

    # Convert feature maps to numpy array
    feature_maps = feature_maps.squeeze().cpu().numpy()

    # Number of feature maps to visualize
    num_feature_maps = feature_maps.shape[0]

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

        ax.imshow(feature_maps[i], cmap='viridis')
        ax.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
