# Standard library & third-party libraries
import json
import pandas as pd
import pickle

# Flask API
from flask import Flask, request
from flask_restful import Api, Resource

# User-defined libraries
from utilities.general_utils import serialize_arf, format_car_specs, format_lead_infos
from utilities.arf_to_dict_conversion import extract_cf_arf, extract_rg_arf
from utilities.arf_training import train_arf_cf, train_arf_rg
import preprocessors.data_loader as data_loader

# Code template: https://github.com/Azure-Samples/azure-sql-db-python-rest-api

app = Flask(__name__)
api = Api(app)

# ------------------------------------------------------------------------------------------------ #
# Load models and data preprocessors
# ------------------------------------------------------------------------------------------------ #

# Load the car price model
with open('models/car_price_model.pkl', 'rb') as f:
    car_price_model = pickle.load(f)
# Load the car price data processor
cp_data_pp = data_loader.load_car_price_data_pp('preprocessors/car_price_data_pp.pkl')

# Load the lead scoring model
with open('models/lead_scoring_model.pkl', 'rb') as f:
    lead_scoring_model = pickle.load(f)
# Load the lead scoring data processor
ls_data_pp = data_loader.load_lead_scoring_data_pp('preprocessors/lead_scoring_data_pp.pkl')

cp_arf_dict = None
ls_arf_dict = None

# ------------------------------------------------------------------------------------------------ #
# Car Price API
# ------------------------------------------------------------------------------------------------ #
class CarPriceTraining(Resource):
    def post(self):
        global cp_arf_dict, car_price_model

        car_inventory_record = request.get_json(force=True)

        # Preprocess car inventory record
        price = car_inventory_record['price']
        car_inventory_record = pd.DataFrame([car_inventory_record])
        car_inventory_record = format_car_specs(car_inventory_record)
        car_inventory_record = cp_data_pp.preprocess(car_inventory_record)

        # Train the car price model
        car_price_model, _ = train_arf_rg(car_price_model, car_inventory_record, [price])
        # Extract the tree weights into a dictionary
        cp_arf_dict = extract_rg_arf(car_price_model, car_inventory_record.columns.tolist())
        # Serialize the dictionary to convert to JSON
        cp_arf_dict_serializable = serialize_arf(cp_arf_dict)

        return cp_arf_dict_serializable, 200

api.add_resource(CarPriceTraining, '/car/training')

# ------------------------------------------------------------------------------------------------ #
# Lead Scoring API
# ------------------------------------------------------------------------------------------------ #

class LeadScoringTraining(Resource):
    def post(self):
        global ls_arf_dict, lead_scoring_model

        lead_info = request.get_json(force=True)

        # Preprocess lead info
        converted = lead_info['converted']
        lead_info = pd.DataFrame([lead_info])
        lead_info = format_lead_infos(lead_info)
        lead_info = ls_data_pp.preprocess(lead_info)
        
        # Train the lead scoring model
        lead_scoring_model, _ = train_arf_cf(lead_scoring_model, lead_info, [converted])
        # Extract the tree weights into a dictionary
        ls_arf_dict = extract_cf_arf(lead_scoring_model, lead_info.columns.tolist())
        # Serialize the dictionary to convert to JSON
        ls_arf_dict_serializable = serialize_arf(ls_arf_dict)

        return ls_arf_dict_serializable, 200

api.add_resource(LeadScoringTraining, '/lead/training')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)