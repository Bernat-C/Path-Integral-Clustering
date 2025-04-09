
from typing import List, Literal
from pydantic import BaseModel

DatasetName = Literal["usps", "mnist", "bc_wisconsin", "synthetic"]

class Config(BaseModel):
    dataset_name: DatasetName
    n_samples: int = None
    n_features: int = None
    target_clusters: int = None
    gaussian_noise_level: float = None
    structural_noise_level: float = None

def load_configs():
    experiment_configs: List[Config] = [
        Config(
            dataset_name = "usps"
        ),
        Config(
            dataset_name = "mnist"
        ),
        Config(
            dataset_name = "bc_wisconsin"
        )
    ]
    
    return experiment_configs