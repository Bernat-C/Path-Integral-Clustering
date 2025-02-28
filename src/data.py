import collections
import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from torchvision.datasets import USPS
from sklearn.datasets import make_blobs

def make_blobs(n_samples=300, n_clusters=3, random_state=42):
    """Generate a simple 2D clustering dataset using make_blobs."""
    X, _ = make_blobs(n_samples=n_samples, n_clusters=n_clusters, random_state=random_state)
    return X

def download_mnist():
    """
    Download the MNIST dataset using kagglehub.
    """
    dataset = kagglehub.dataset_download("hojjatk/mnist-dataset")
    return dataset

def load_usps():
    usps_dataset = USPS(root="./data/USPS", train=True, download=True)
    usps_dataset_test = USPS(root="./data/USPS", train=False, download=True)
    
    images = np.array([np.array(img[0]) for img in usps_dataset] + [np.array(img[0]) for img in usps_dataset_test])
    labels = np.array([img[1] for img in usps_dataset] + [img[1] for img in usps_dataset_test])
    
    print(len(images))
    print(collections.Counter(labels).items())
    print(len(set(labels)))
    
    return images, labels

def load_mnist():
    """
    Select all images of digits from 0 to 4 in the MNIST test set.
    Should result in:
    - 5139 samples
    - 5 clusters
    - 980 min cluster size
    - 1135 max cluster size
    - 784 dimensionality
    """
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    mask = np.isin(y_test, [0, 1, 2, 3, 4])
    images, labels = x_test[mask], y_test[mask]
    
    assert(len(images == 5139))
    assert(len(set(labels)) == 5)
    
    return images, labels

if __name__ == "__main__":
    images_x, image_y = load_mnist()

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images_x[i], cmap="gray")
        plt.title(f"Label: {images_x[i]}")
        plt.axis("off")
    plt.show()
