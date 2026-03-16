
import json
import pandas as pd
from datetime import datetime
import pickle
from sklearn.preprocessing import RobustScaler
timestamp = datetime.now().strftime("%m-%d_%H-%M")


SAMPLE_INPUT = 'test_data.csv'  # path to CSV file with input data for prediction
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
FEATURES_PATH = "model/random_forest_regressor_features.json"  # path to JSON file with list of features used in model
MODEL_PATH = "model/random_forest_regressor.pkl"  # path to pickle file with trained model
# before adding in there were 18 features before merging on zip - led to 43 after merging on zip (18 + 33)
SALES_COLUMN_SELECTION = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

def load_model():
    """Load a trained model from a pickle file."""
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
    features = json.load(open(FEATURES_PATH, 'r'))
    print(f"Model loaded successfully with features: {features}")
    print("Model loaded successfully")
    return loaded_model, features
    

def preprocess_input(input_path:str) -> pd.DataFrame:
    """Preprocess input data by merging with demographics and selecting features."""
    input_df = pd.read_csv(input_path, 
                           usecols=SALES_COLUMN_SELECTION,
                           dtype={'zipcode': str})
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
  
    print('input features length before merge:', len(input_df.columns))
    print('features json length', len(json.load(open(FEATURES_PATH, 'r'))))

    merged_input = input_df.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    print('input features length after merge:', len(merged_input.columns))
    return merged_input

def predict(input_data_path: str) -> float:

    loaded_model, features = load_model()
    # Run inference on future data and save predictions
    input_df = preprocess_input(input_data_path)

    ########## DETERMINE IF NEED ##########
    #verify input data has the same features as the model was trained on 
    missing_features = set(features) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Input data is missing the following features required by the model: {missing_features}")
    

    # reorder input data to match the order of features used in training
    input_df = input_df[features]

    future_pred = loaded_model.predict(input_df)

    return float(future_pred[0])


if __name__ == "__main__":
    prediction = predict(SAMPLE_INPUT)
    print(f"Predicted price: {prediction:.2f}")