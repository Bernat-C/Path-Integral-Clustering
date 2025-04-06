import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Step 1: Generate the original dataset using make_blobs
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

# Function to add Gaussian noise
def add_gaussian_noise(X, noise_level=0.1):
    # Generate Gaussian noise with mean 0 and specified standard deviation
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    # Add noise to the original dataset
    X_noisy = X + noise
    return X_noisy

# Step 2: Add noise with different noise levels and visualize
noise_levels = [1, 5, 10, 100]  # Different levels of Gaussian noise
plt.figure(figsize=(12, 8))

for idx, noise_level in enumerate(noise_levels, start=1):
    X_noisy = add_gaussian_noise(X, noise_level)
    
    plt.subplot(2, 2, idx)
    plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y, cmap=plt.cm.Paired, s=10)
    plt.title(f"Gaussian Noise Level: {noise_level}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
