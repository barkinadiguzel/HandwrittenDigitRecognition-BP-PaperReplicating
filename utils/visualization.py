import matplotlib.pyplot as plt
import torch

def show_digit(image, label=None):
    """
    Show a single digit image.
    image: [1,H,W] torch tensor or [H,W] numpy array
    label: optional label
    """
    if isinstance(image, torch.Tensor):
        image = image.squeeze().numpy()  # [H,W]
    
    plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
    if label is not None:
        plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

def show_feature_map(feature_map):
    """
    Show feature map (single channel)
    feature_map: [C,H,W] torch tensor or [H,W] numpy array
    """
    if isinstance(feature_map, torch.Tensor):
        if feature_map.ndim == 3:
            # show first channel by default
            feature_map = feature_map[0].detach().numpy()
        else:
            feature_map = feature_map.detach().numpy()
    
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')
    plt.show()
