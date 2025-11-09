# This script is a quick check to ensure the USPS dataset loads and visualizes correctly
import bz2
import numpy as np
import matplotlib.pyplot as plt

# Load the compressed dataset
file_path = "./data/train/usps.bz2"
with bz2.open(file_path, "rt") as f:
    lines = f.readlines()

# Take the first line (one image sample)
line = lines[0].strip()
parts = line.split()

# First part is the label
label = int(parts[0])

# Remaining parts are pixel values in "index:value" format
pixels = np.array([float(p.split(":")[1]) for p in parts[1:]])

# Rescale from [-1,1] to [0,255]
pixels = ((pixels + 1) * 127.5).astype(np.uint8)

# Reshape to 16x16 image
img = pixels.reshape(16,16)

# Visualize the image
plt.imshow(img, cmap='gray')
plt.title(f"Label: {label}")
plt.show()
