import h5py
import torch

class USPSH5Dataset(torch.utils.data.Dataset):
    """USPS dataset loader with preprocessing: reshape + normalize to [-1,1]"""
    def __init__(self, file_path, split="train"):
        super().__init__()
        with h5py.File(file_path, "r") as f:
            self.images = torch.tensor(f[split]["data"][:], dtype=torch.float32)  # [N, H*W]
            self.labels = torch.tensor(f[split]["target"][:], dtype=torch.long)

        # reshape from [N, H*W] to [N, 1, H, W] (grayscale)
        N = self.images.shape[0]
        self.images = self.images.view(N, 1, 16, 16)

        # normalize to [-1, 1] (originally 0–255)
        self.images = self.images / 127.5 - 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# test visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    file_path = "../data/usps.h5"
    dataset = USPSH5Dataset(file_path, split="train")
    print("Train size:", len(dataset))

    img, label = dataset[0]
    print("Shape:", img.shape, "Label:", label.item())

    plt.imshow(img.squeeze().numpy(), cmap="gray")
    plt.title(f"Label: {label.item()}")
    plt.axis("off")
    plt.show()
