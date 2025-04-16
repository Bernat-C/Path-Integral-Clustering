import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import USPS
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.datasets import load_breast_cancer

def generate_synthetic(n_samples, n_features, centers, ds_type, random_state=None):
    """Generate a simple 2D clustering dataset using make_blobs."""
    if ds_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    elif ds_type == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.05, random_state=random_state)
    else:
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return X, y

def add_gaussian_noise(X, noise_level):
    """ Add Gaussian noise to the dataset. """
    # Generate Gaussian noise with mean 0 noise_level standard deviation
    noise = np.random.normal(loc=0.0, scale=noise_level/10, size=X.shape)
    return X + noise

def add_structural_noise(X, y_true, noise_level=0.1):
    """ Add structural noise to the dataset by removing a percentage of points. """
    num_points_to_remove = int(len(X) * noise_level)
    
    # Randomly select indices of points to remove
    indices_to_remove = np.random.choice(X.shape[0], num_points_to_remove, replace=False)
    X_noisy = np.delete(X, indices_to_remove, axis=0)
    y_noisy = np.delete(y_true, indices_to_remove, axis=0)
        
    return X_noisy, y_noisy

def load_usps():
    """ Load the USPS dataset from torchvision.datasets """
    usps_dataset = USPS(root="./data/USPS", train=True, download=True)
    usps_dataset_test = USPS(root="./data/USPS", train=False, download=True)
    
    images = np.array([np.array(img[0]) for img in usps_dataset] + [np.array(img[0]) for img in usps_dataset_test])
    labels = np.array([img[1] for img in usps_dataset] + [img[1] for img in usps_dataset_test])
    
    print(len(images))
    print(len(set(labels)))
    
    images = images.reshape(-1,16*16)
    
    print(images.shape)
    
    return images, labels

def load_mnist():
    """
    Load the MNIST dataset from tensorflow.keras.datasets, selects all images of digits from 0 to 4 in the MNIST test set.
    Should result in:
    - 5139 samples
    - 5 clusters
    - 980 min cluster size
    - 1135 max cluster size
    - 784 dimensionality
    """
    from tensorflow.keras.datasets import mnist
    _, (x_test, y_test) = mnist.load_data()
    
    mask = np.isin(y_test, [0, 1, 2, 3, 4])
    images, labels = x_test[mask], y_test[mask]
    
    assert(len(images == 5139))
    assert(len(set(labels)) == 5)
    
    images = images.reshape(-1,28*28)
    
    return images, labels

def load_bc_wisconsin():
    """ Load the Breast Cancer Wisconsin dataset from sklearn.datasets """
    data = load_breast_cancer()
        
    return data.data, data.target

def analyze_ds(name, images_x, image_y):
    """ Analyze the dataset """
    
    print("##########################")
    print(f"Analyzing {name} dataset")
    print("##########################")
    
    print(images_x.shape)
    print(len(set(image_y)))
    print(min([len(np.where(image_y == i)[0]) for i in set(image_y)]))
    print(max([len(np.where(image_y == i)[0]) for i in set(image_y)]))

    if name != "BC-Wisconsin":
        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            side = int(np.sqrt(images_x.shape[1]))
            img = images_x[i].reshape(side,side)
            plt.imshow(img, cmap="gray")
            plt.title(f"Label: {images_x[i]}")
            plt.axis("off")
        plt.show()
    
if __name__ == "__main__":
    images_x, image_y = load_usps()
    analyze_ds("USPS", images_x, image_y)
    
    images_x, image_y = load_mnist()
    analyze_ds("MNIST", images_x, image_y)
    
    images_x, image_y = load_bc_wisconsin()
    analyze_ds("BC-Wisconsin", images_x, image_y)
