# Path Integral Clustering

This repository implements the Path Integral Clustering Algorithm and various clustering algorithms, and evaluates their performance on synthetic and real-world datasets. The project also includes experiments to analyze the robustness of clustering algorithms under different types of noise.

The implementation of the Path Integral Clustering algorithm is based on "Agglomerative clustering via maximum incremental path integral" (2013).

## Features

- **Clustering Algorithms**:
  - Path Integral Clustering (PIC)
  - Affinity Propagation (AP)
  - Average Linkage (A-Link)
  - Single Linkage (S-Link)
  - Complete Linkage (C-Link)
  - Zeta Function Clustering (Zell)
  - Diffusion Kernel Clustering (D-kernel)

- **Datasets**:
  - Synthetic datasets: `moons`, `blobs`, `circles`
  - Real-world datasets: MNIST, USPS, Breast Cancer Wisconsin

- **Noise Experiments**:
  - Gaussian noise
  - Structural noise

- **Evaluation Metrics**:
  - Normalized Mutual Information (NMI)
  - Clustering Error (CE)
  - Silhouette Score (Silhouette)
  - Davies-Bouldin Index (DBI)
  - Calinski-Harabasz Index (CH)

- **Visualization**:
  - Cluster visualizations
  - Noise experiment results

## Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

1. **Single Experiment**:
   Run a single clustering experiment on a synthetic dataset:
   ```bash
   python src/main.py
   ```
   By default, the script runs on the `moons` dataset with 500 samples and 2 features.

2. **Noise Experiments**:
   Run noise experiments on synthetic datasets (`moons`, `blobs`, `circles`):

   Modify src/main.py with the appropiate configuration.
   ```bash
   python src/main.py
   ```
   The script evaluates clustering performance under varying levels of Gaussian and structural noise.

3. **Real Dataset Experiments**:
   Add configurations for real datasets in the `configs` directory and run:
   ```bash
   python src/main.py
   ```

### Configuration

The clustering experiments are configured using the `Config` class. Example configuration:
```python
config = Config(
    name="synthetic",
    dataset_name="moons",
    n_samples=500,
    n_features=2,
    target_clusters=2
)
```

### Results

- Results for each experiment are saved as CSV files in the data directory.
- Visualizations are saved in the plots directory.

## Project Structure

```
Path-Integral-Clustering/
├── src/
│   ├── main.py                     # Main script for running experiments
│   ├── config.py                   # Configuration management
│   ├── data.py                     # Dataset loading and generation
│   ├── algorithms.py               # Other clustering algorithm implementations
│   ├── metrics.py                  # Evaluation metrics
│   ├── nearest_neighbour_init.py   # Nearest neighbour initialization algorithm
│   ├── path_integral.py            # Computation of the path integral as described in the paper
│   ├── pic.py                      # Path Integral Clustering algorithm
│   ├── utils.py                    # Utility functions
│   ├── visualize.py                # Visualization utilities
├── data/                           # Experiment results
│   ├── plots/                      # Generated plots
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Dependencies

- Python 3.8+
- Required Python packages are listed in requirements.txt.
