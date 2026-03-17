from flask import Flask, request
from flask_restful import Resource, Api

#import inference model and preprocessing from run_model.py
from model import run_inference

app = Flask(__name__)
api = Api(app)


class PredictPrice(Resource):
    def post(self):
        data = request.get_json()
        if not data:
            return {"error" : "No input data provided"}, 400
        try:
            predictions= run_inference(data)
            return {"predicted_price": predictions}, 200
        except Exception as e:
            return {"error": str(e)}, 422
    

api.add_resource(PredictPrice, '/predict')

if __name__ == "__main__":
    app.run(debug=True)