
import json
import pandas as pd
from datetime import datetime
import pickle
from sklearn.preprocessing import RobustScaler
timestamp = datetime.now().strftime("%m-%d_%H-%M")


#SAMPLE_INPUT = 'test_data.json'  # path to JSON file with input data for prediction
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
FEATURES_PATH = "model/random_forest_regressor_features.json"  # path to JSON file with list of features used in model
MODEL_PATH = "model/random_forest_regressor.pkl"  # path to pickle file with trained model
# before adding in there were 18 features before merging on zip - led to 43 after merging on zip (18 + 33)
SALES_COLUMN_SELECTION = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

## need to receive JSON POST data

def load_model():
    """Load a trained model from a pickle file."""
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
    features = json.load(open(FEATURES_PATH, 'r'))
    print(f"Model loaded successfully with features: {features}")
    print("Model loaded successfully")
    return loaded_model, features
    

def preprocess_input(input_data) -> pd.DataFrame:
    """Merge demographics with input JSON data"""
    # with open(input_data, 'r') as f:
    #     input_data = json.load(f)
    if isinstance(input_data, dict):
        input_data_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data_df = pd.DataFrame(input_data)
    else:
        raise ValueError("Input data must be a list of dictionaries or a single dictionary")

    input_df = input_data_df[SALES_COLUMN_SELECTION]  # select only the columns used in training
    input_df.loc[:,'zipcode'] = input_df['zipcode'].astype(int).astype(str)  # ensure zipcode is string for merging
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
  
    # print('input features length before merge:', len(input_df.columns))
    # print('features json length', len(json.load(open(FEATURES_PATH, 'r'))))

    merged_input = input_df.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    # print('input features length after merge:', len(merged_input.columns))
    return merged_input

def run_inference(input_data) -> float:
    '''
    Run inference on JSON input data
    Args:
        input_data: list of dicts
    Returns:
        predicted prices as a floats
    '''

    loaded_model, features = load_model()
    # Run inference on future data and save predictions
    input_df = preprocess_input(input_data)

    ########## DETERMINE IF NEED ##########
    #verify input data has the same features as the model was trained on 
    missing_features = set(features) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Input data is missing the following features required by the model: {missing_features}")
    

    # reorder input data to match the order of features used in training
    input_df = input_df[features]
    future_preds = loaded_model.predict(input_df)
    return [float(p) for p in future_preds]


# if __name__ == "__main__":
#     prediction = run_inference(SAMPLE_INPUT)
#     for i, pred in enumerate(prediction):
#         print(f"Predicted price for input {i+1}: {pred:.2f}")
    