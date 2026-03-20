import itertools
import json
import os
import pathlib
import pickle
from typing import List
from typing import Tuple
import comet_ml
import pandas
import numpy as np
from datetime import datetime
timestamp = datetime.now().strftime("%m-%d_%H-%M")

from dotenv import load_dotenv
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing


from MetricsClass import Metrics_Summary
from ModelClass import models

load_dotenv()
api_key = os.getenv("API_KEY")

SALES_PATH = "train/data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "app/data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

def add_features(df: pandas.DataFrame) -> pandas.DataFrame:

    # # home renovation features
    # current_year = 2015
    # df['house_age'] = current_year - df['yr_built']
    # df['was_renovated'] = (df['yr_renovated']>0).astype(int)
    # df['years_since_renovation'] = np.where(
    #     df['yr_renovated'] > 0, 
    #     current_year - df['yr_renovated'], 
    #     df['house_age']
    # )

    # space ratio
    df['lot_to_living_ratio'] = df['sqft_lot'] / df['sqft_living']
    df['above_to_living_ratio'] = df['sqft_above'] / df['sqft_living']
    df['basement_present'] = (df['sqft_basement'] > 0).astype(int)
    
    df['sqft_per_bedroom'] = df['sqft_living'] / (df['bedrooms'].replace(0, 1))  # avoid division by zero
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'].replace(0, 1))  # avoid division by zero

    return df

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with demographics data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")

    #add in two features
    merged_data = add_features(merged_data)

    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(x, y,
                                                                          train_size=0.8,
                                                                          test_size=0.2, 
                                                                          random_state=42)

    # model = pipeline.make_pipeline(preprocessing.RobustScaler(),
    #                                neighbors.KNeighborsRegressor()).fit(
    #                                    x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    for config in models:
        if config.param_grid:
            keys = config.param_grid.keys()
            param_combinations = [
                dict(zip(keys, values)) for values in itertools.product(*config.param_grid.values())
            ]
        else:
            param_combinations = [config.params]
        
        for params in param_combinations:
            experiment = comet_ml.start(
                project_name="housing-model",
                workspace="kayla-rossi",
                api_key=api_key,
            )
            param_str = "_".join(f"{key}_{value}" for key, value in params.items())
            experiment.set_name(f"{config.name}_{param_str}_{timestamp}")


            if config.needs_scaling:
                model = pipeline.make_pipeline(
                    preprocessing.RobustScaler(), 
                    config.model.set_params(**params))
            else:
                model = config.model.set_params(**params)

            model.fit(x_train, y_train)

            y_train_preds = model.predict(x_train)
            y_test_preds = model.predict(_x_test)

            #metrics for training data
            train_metrics = Metrics_Summary(y_train, y_train_preds, config.name)
            train_metrics_dict = train_metrics.as_dict()
            for name, value in train_metrics_dict.items():
                experiment.log_metric(f"train_{name}", value)
            print("Training Metrics:")
            train_metrics.print_summary()

            # metrics for test data
            test_metrics = Metrics_Summary(_y_test, y_test_preds, config.name)
            test_metrics_dict = test_metrics.as_dict()
            for name, value in test_metrics_dict.items():
                experiment.log_metric(f"test_{name}", value)
            print("Test Metrics:")
            test_metrics.print_summary()

            # Output model artifacts: pickled model and JSON list of features
            run_name = f"{config.name}_{param_str}"
            pickle.dump(model, open(output_dir / f"{run_name}.pkl", 'wb'))
            json.dump(list(x_train.columns),
                    open(output_dir / f"{run_name}_features.json", 'w'))

            experiment.log_parameters(params)
            # experiment.log_asset(file_data = str(output_dir / f"{run_name}_features.json"), 
            #                       file_name="features", 
            #                       metadata={"description": "List of features used in the model"})
            experiment.log_model(name=run_name, 
                                file_or_folder=str(output_dir / f"{run_name}.pkl"), 
                                metadata={"features": list(x_train.columns)})
            
            experiment.end()


if __name__ == "__main__":
    main()
