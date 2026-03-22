
import json
import pandas as pd
import pickle
import logging
#logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gunicorn.error')

# SAMPLE_INPUT = 'test/single_test_data.json'  # path to JSON file with input data for prediction
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
FEATURES_PATH = "model_features/gradient_boosting_n_estimators_100_learning_rate_0.1_max_depth_3_loss_huber_features.json"  # path to JSON file with list of features used in model
MODEL_PATH = "model_features/gradient_boosting_n_estimators_100_learning_rate_0.1_max_depth_3_loss_huber.pkl"  # path to pickle file with trained model

## need to receive JSON POST data

def load_model():
    """
    Load a trained machine learning model and its feature list from disk.

    Returns:
        tuple: (loaded_model, features)
            loaded_model: The trained model loaded from a pickle file.
            features: List of feature names used during model training.

    Raises:
        ValueError: If loading the model or features fails.
    """
    try:
        loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
        features = json.load(open(FEATURES_PATH, 'r'))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or features: {str(e)}")
        raise ValueError(f"Error loading model or features: {str(e)}")
    return loaded_model, features
    
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the input DataFrame for model inference.

    Args:
        df (pd.DataFrame): Input DataFrame with raw features.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    # space ratio
    df['lot_to_living_ratio'] = df['sqft_lot'] / df['sqft_living'].replace(0, 1)  # avoid division by zero
    df['above_to_living_ratio'] = df['sqft_above'] / df['sqft_living'].replace(0, 1)  # avoid division by zero
    df['basement_present'] = (df['sqft_basement'] > 0).astype(int)
    df['sqft_per_bedroom'] = df['sqft_living'] / (df['bedrooms'].replace(0, 1))  # avoid division by zero
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'].replace(0, 1))  # avoid division by zero
    return df

def preprocess_input(input_data, features) -> pd.DataFrame:
    """
    Merge demographics with input JSON data, engineer features, and select columns to match model training.

    Args:
        input_data (dict or list): Input data as a dictionary or list of dictionaries.
        features (list): List of feature names expected by the model.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for model inference.

    Raises:
        ValueError: If input data is missing required features or cannot be processed.
    """
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

    # add in additional features
    merged_input = add_features(merged_input)

    #verify input data has the same features as the model was trained on 
    missing_features = set(features) - set(merged_input.columns)
    if missing_features:
        logger.error(f"Input data is missing the following features required by the model: {missing_features}")
        raise ValueError(f"Input data is missing the following features required by the model: {missing_features}")
    
    merged_input = merged_input[features]  # down select to features used in training
    logger.info("Input data preprocessed successfully")
    return merged_input

def run_inference(input_data) -> float:
    """
    Run inference on input data and return formatted price predictions.

    Args:
        input_data (dict or list): Input data as a dictionary or list of dictionaries.

    Returns:
        list: List of formatted price prediction strings for each input record.

    Raises:
        ValueError: If inference fails or input data is invalid.
    """
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