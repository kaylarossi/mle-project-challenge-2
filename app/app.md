# app.py Documentation

This module implements a Flask REST API for serving house price predictions using a trained machine learning model. It exposes endpoints for making predictions with JSON input data.

## Endpoints

### POST /predict
- **Description:**
  - Accepts a JSON array of records, each containing the required features for prediction.
  - Returns a list of predicted prices for each input record.
- **Request Body:**
  - JSON array of objects, each with the following fields and more as they exist in the training dataset:
    - bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, zipcode
- **Response:**
  - 200: `{ "predicted_price(s)": [ ... ] }`
  - 400: `{ "error": "No input data provided" }`
  - 422: `{ "error": "Error during inference: ..." }`

### POST /predict/simple
- **Description:**
  - Accepts a JSON object or array with only the required columns. If extra columns are present, returns an error.
  - Returns a list of predicted prices for each input record.
- **Request Body:**
  - JSON object or array with only the required fields:
    - bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, zipcode
- **Response:**
  - 200: `{ "predicted_price(s)": [ ... ] }`
  - 400: `{ "error": "No input data provided" }` or `{ "error": "Extra columns not allowed: ..." }`
  - 422: `{ "error": "Error during inference: ..." }`

## Implementation Details
- Uses Flask and Flask-RESTful for API structure.
- Imports `run_inference` from model.py to perform predictions.
- Configures logging for both Flask and Gunicorn environments.
- Handles and logs errors, returning appropriate HTTP status codes and messages.

## Example Usage

**Request:**
POST /predict
```json
[
  {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1,
    "sqft_above": 1800,
    "sqft_basement": 0,
    "zipcode": "98103"
  }
]
```
**Response:**
```json
{
  "predicted_price(s)": ["Price for house 1: $650,000.00"]
}
```

---

For more details, see the code comments and logging output.
