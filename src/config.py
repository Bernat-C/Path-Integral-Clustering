
from typing import List, Literal
from pydantic import BaseModel

DatasetName = Literal["usps", "mnist", "bc_wisconsin", "synthetic"]

class Config(BaseModel):
    name: str
    dataset_name: DatasetName
    n_samples: int = None
    n_features: int = None
    target_clusters: int = None
    gaussian_noise_level: float = None
    structural_noise_level: float = None

def load_configs():
    experiment_configs: List[Config] = [
        Config(
            name = "usps",
            dataset_name = "usps",
            target_clusters=10
        ),
        Config(
            name = "mnist",
            dataset_name = "mnist",
            target_clusters=5
        ),
        Config(
            name = "bc_wisconsin",
            dataset_name = "bc_wisconsin",
            target_clusters=2
        )
    ]
    
    return experiment_configs