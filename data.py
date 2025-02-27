import collections
import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from torchvision.datasets import USPS

def download_mnist():
    """
    Download the MNIST dataset using kagglehub.
    """
    dataset = kagglehub.dataset_download("hojjatk/mnist-dataset")
    return dataset

def get_usps_images():
    usps_dataset = USPS(root="./data/USPS", train=True, download=True)
    usps_dataset_test = USPS(root="./data/USPS", train=False, download=True)
    
    images = np.array([np.array(img[0]) for img in usps_dataset] + [np.array(img[0]) for img in usps_dataset_test])
    labels = np.array([img[1] for img in usps_dataset] + [img[1] for img in usps_dataset_test])
    
    print(len(images))
    print(collections.Counter(labels).items())
    print(len(set(labels)))
    
    return images, labels

def get_caltech_images():
    """
    Load images from a folder where each subfolder represents a class.
    """
    folder_path = r"./data/Caltech256"
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # Get sorted class names
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                try:
                    with Image.open(image_path) as img:
                        images.append(np.array(img.convert("RGB")))  # Convert to RGB
                        labels.append(class_dict[class_name])
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    
    print("First we have to extract local image features (which we are not going to do)")
    return np.array(images), np.array(labels)

def get_mnist_images():
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
    images_x, image_y = get_mnist_images()

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images_x[i], cmap="gray")
        plt.title(f"Label: {images_x[i]}")
        plt.axis("off")
    plt.show()
