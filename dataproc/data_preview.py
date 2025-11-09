import h5py
import matplotlib.pyplot as plt
import torch

def show_image(image, label=None):
    """Visualize a single digit image"""
    if isinstance(image, torch.Tensor):
        image = image.squeeze().numpy()
    plt.imshow(image, cmap='gray')
    if label is not None:
        plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    file_path = "../data/usps.h5"
    with h5py.File(file_path, "r") as f:
        images = torch.tensor(f["train"]["data"][:], dtype=torch.float32)
        labels = torch.tensor(f["train"]["target"][:], dtype=torch.long)

    N = images.shape[0]
    images = images.view(N, 1, 16, 16)  # [N,1,16,16]

    if images.max() > 1.5:
        images = images / 255.0

    img, label = images[0], labels[0]
    print("Shape:", img.shape, "Label:", label.item())
    show_image(img, label.item())
