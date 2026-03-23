from dataclasses import dataclass
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

@dataclass
class ModelConfig:
    """Configuration for training a model."""
    name: str
    model: object
    needs_scaling: bool = False
    features: list = None
    target: str = None
    params: dict = None
    param_grid: dict = None
    
models = [
    ModelConfig(
        name="knn_regressor",
        model=neighbors.KNeighborsRegressor(),
        needs_scaling=True,
        features=[...],  # list of feature names
        target="price",
        params={"n_neighbors": 7, "weights": "distance"},
        param_grid={"n_neighbors": [5, 7, 9]}  
    ),
    ModelConfig(
        name="random_forest_regressor",
        model=RandomForestRegressor(),
        needs_scaling=False,
        features=[...],  # list of feature names
        target="price",
        params={"n_estimators": 150, "max_depth": None, "criterion": "absolute_error"},
        param_grid={"n_estimators": [100,150, 200], "max_depth": [None, 5, 10]}  
    ),
    ModelConfig(
        name = "gradient_boosting",
        model = GradientBoostingRegressor(),
        needs_scaling = False,
        features = [...],  # list of feature names
        target = "price",
        params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "loss": "huber"},
        param_grid = {"loss": ["absolute_error", "huber"], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]} #"n_estimators": [100, 150, 200], 
    )
]


