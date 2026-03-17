from dataclasses import dataclass
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor

@dataclass
class ModelConfig:
    """Configuration for training a model."""
    name: str
    model: object
    needs_scaling: bool = False
    features: list = None
    target: str = None
    params: dict = None
    
models = [
    ModelConfig(
        name="knn_regressor",
        model=neighbors.KNeighborsRegressor(),
        needs_scaling=True,
        features=[...],  # list of feature names
        target="price",
        params={"n_neighbors": 5}  
    ),
    ModelConfig(
        name="random_forest_regressor",
        model=RandomForestRegressor(n_estimators=100),
        needs_scaling=False,
        features=[...],  # list of feature names
        target="price",
        params={"n_estimators": 100}  
    )
]


