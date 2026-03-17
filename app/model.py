
import json
import pandas as pd
import pickle
import logging
#logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gunicorn.error')

# SAMPLE_INPUT = 'test/single_test_data.json'  # path to JSON file with input data for prediction
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
FEATURES_PATH = "model/random_forest_regressor_features.json"  # path to JSON file with list of features used in model
MODEL_PATH = "model/random_forest_regressor.pkl"  # path to pickle file with trained model

## need to receive JSON POST data

def load_model():
    """Load a trained model from a pickle file."""
    try:
        loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
        features = json.load(open(FEATURES_PATH, 'r'))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or features: {str(e)}")
        raise ValueError(f"Error loading model or features: {str(e)}")
    return loaded_model, features
    

def preprocess_input(input_data, features) -> pd.DataFrame:
    """Merge demographics with input JSON data and down select features to match those used in training the model."""
    # with open(input_data, 'r') as f:
    #     input_data = json.load(f)
    try:
        if isinstance(input_data, dict):
            input_data_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_data_df = pd.DataFrame(input_data)
    except Exception as e:
        logger.error(f"Error processing input data: {str(e)}")
        raise ValueError(f"Error processing input data: {str(e)}")
    
    input_data_df.loc[:,'zipcode'] = input_data_df['zipcode'].astype(int).astype(str)  # ensure zipcode is string for merging
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})

    merged_input = input_data_df.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    #verify input data has the same features as the model was trained on 
    missing_features = set(features) - set(merged_input.columns)
    if missing_features:
        logger.error(f"Input data is missing the following features required by the model: {missing_features}")
        raise ValueError(f"Input data is missing the following features required by the model: {missing_features}")
    
    merged_input = merged_input[features]  # down select to features used in training
    logger.info("Input data preprocessed successfully")
    return merged_input

def run_inference(input_data) -> float:
    '''
    Run inference on JSON input data
    Args:
        input_data: list of dicts
    Returns:
        predicted prices as a floats formatted as prices
    '''
    loaded_model, features = load_model()
    # Run inference on future data and save predictions
    input_df = preprocess_input(input_data, features)
    # reorder input data to match the order of features used in training
    input_df = input_df[features]
    try:
        future_preds = loaded_model.predict(input_df)
        output = [f"Price for house {i+1}: ${float(p):,.2f}" for i, p in enumerate(future_preds)]
        logger.info("Inference ran successfully")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise ValueError(f"Error during inference: {str(e)}")
    
    return output

# if __name__ == "__main__":
    # prediction = run_inference(SAMPLE_INPUT)
    # print(prediction)