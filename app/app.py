from flask import Flask, request
from flask_restful import Resource, Api
import logging

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#import inference model and preprocessing from run_model.py
from model import run_inference

app = Flask(__name__)
api = Api(app)

#configure logging for gunicorn
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

REQUIRED_COLUMNS = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

class PredictPrice(Resource):
    def post(self):
        data = request.get_json()
        if not data:
            app.logger.warning("No input data provided in the request.")
            return {"error": "No input data provided"}, 400
        try:
            predictions= run_inference(data)
            app.logger.info(f"Predictions made successfully: {len(predictions)} record(s) processed.")
            return {"predicted_price(s)": predictions}, 200
        except ValueError as e:
            app.logger.error(f"Error during inference: {str(e)}")
            return {"error": str(e)}, 422
        
class PredictPriceSimple(Resource):
    def post(self):
        data = request.get_json()   
        if not data:
            app.logger.warning("No input data provided in the request.")
            return {"error": "No input data provided"}, 400
        record = data[0] if isinstance(data, list) else data
        extra_columns = set(record.keys()) - set(REQUIRED_COLUMNS)
        if extra_columns:
            app.logger.warning(f"Received extra columns: {extra_columns}, user redirected to use /predict endpoint.")
            return {"error": f"Extra columns not allowed: {extra_columns}. Use /predict endpoint."}, 400
        try:
            predictions= run_inference(data)
            app.logger.info(f"Predictions made successfully: {len(predictions)} record(s) processed.")
            return {"predicted_price(s)": predictions}, 200
        except ValueError as e:
            app.logger.error(f"Error during inference: {str(e)}")
            return {"error": str(e)}, 422
    

api.add_resource(PredictPrice, '/predict')
api.add_resource(PredictPriceSimple, '/predict/simple')

if __name__ == "__main__":
    app.run(debug=True)