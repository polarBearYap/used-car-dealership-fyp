import os
import json
import pyodbc
import socket
from flask import Flask, request
from flask_restful import Api, Resource
from threading import Lock
from tenacity import *
import logging
import numpy as np
import pandas as pd
import requests

# User-defined libraries
import preprocessors.data_loader as data_loader
from utilities.general_utils import deserialize_arf
from explainers.car_price import CarPriceExplainer
from explainers.lead_scoring import LeadScoringExplainer
from drift_detectors.car_price import CarPriceDriftDetector
from drift_detectors.lead_scoring import LeadScoringDriftDetector

# Code template: https://github.com/Azure-Samples/azure-sql-db-python-rest-api

app = Flask(__name__)
api = Api(app)

# ----------------------------------------------------------------------------------- #
# Load the car price preprocessor and train set
# ----------------------------------------------------------------------------------- #
# Load the car price data processor and the test subsample
cp_data_pp, cp_X_test_truth_av_subsample = data_loader.load_car_price_data('preprocessors/car_price_data_pp.pkl')

car_train = pd.read_csv(f'data/cp_train_set.csv')

# Shuffle the car records
rng = np.random.default_rng(2022)
car_train = car_train.reindex(rng.permutation(car_train.index))
car_train = car_train.reset_index(drop=True)

# Remove columns that are not needed to perform inference
car_train = car_train.copy().drop(columns=['model'], axis=1)

cp_target_attr = 'price'
cp_X_train = car_train.drop(columns=cp_target_attr, axis=1)
cp_y_train = car_train[cp_target_attr]

# Preprocess the data
cp_X_train  = cp_data_pp.preprocess(cp_X_train)

# Randomly choose 100 samples from the train set
rng = np.random.default_rng(2022)
idx_arr = rng.choice(range(len(cp_X_train)), 100)
cp_X_train_subsample = cp_X_train.iloc[idx_arr, :].copy()
# ----------------------------------------------------------------------------------- #
# END
# ----------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------- #
# Load the lead scoring train set
# ----------------------------------------------------------------------------------- #
# Load the lead scoring data processor and the test subsample
ls_data_pp, ls_X_test_truth_av_subsample = data_loader.load_lead_scoring_data('preprocessors/lead_scoring_data_pp.pkl')

ls_train = pd.read_csv(f'data/ls_train_set.csv')

ls_target_attr = 'converted'
ls_X_train = ls_train.drop(columns=ls_target_attr, axis=1)
ls_y_train = ls_train[ls_target_attr]

ls_X_train = ls_data_pp.preprocess(ls_X_train)

# Randomly choose 100 samples from the train set
rng = np.random.default_rng(2022)
idx_arr = rng.choice(range(len(ls_X_train)), 100)
ls_X_train_subsample = ls_X_train.iloc[idx_arr, :].copy()
# ----------------------------------------------------------------------------------- #
# END
# ----------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------- #
# Load models, explainers, and drift detectors
# ----------------------------------------------------------------------------------- #

# Load the car price model
with open('models/car_price_model_dict.json', 'r') as f:
    cp_arf_dict_serializable = json.load(f)
cp_arf_dict = deserialize_arf(cp_arf_dict_serializable)

# Initialize wrapper class for car price explainer
car_price_explainer = CarPriceExplainer(
    cp_data_pp, 
    cp_arf_dict, 
    cp_X_train,
    cp_y_train,
    cp_X_train_subsample,
    cp_X_test_truth_av_subsample
)

# Load the lead scoring model
with open('models/lead_scoring_model_dict.json', 'r') as f:
    ls_arf_dict_serializable = json.load(f)
ls_arf_dict = deserialize_arf(ls_arf_dict_serializable)

# Initialize wrapper class for lead scoring explainer
lead_scoring_explainer = LeadScoringExplainer(
    ls_data_pp, 
    ls_arf_dict,
    ls_X_train, 
    ls_y_train,
    ls_X_train_subsample, 
    ls_X_test_truth_av_subsample
)

# Initialize wrapper class for car price drift detector
# `doors` and `seat_capacity` are considered as ordinal features, 
# thus they are considered as categorical features
cp_X_train_ = cp_X_train.copy().astype({'doors': 'object', 'seat_capacity': 'object'})

car_price_drift_detector = CarPriceDriftDetector(
    cp_X_train_, 
    cp_data_pp
)

# Initialize wrapper class for lead scoring drift detector
lead_scoring_drift_detector = LeadScoringDriftDetector(
    ls_X_train, 
    ls_data_pp
)

# ------------------------------------------------------------------------------------------------ #
# Database Connection Manager (By Microsoft)
# ------------------------------------------------------------------------------------------------ #

class ConnectionManager(object):
    """
    Implement singleton to avoid global objects

    Full credit: https://github.com/Azure-Samples/azure-sql-db-python-rest-api/blob/master/app.py
    """
    __instance = None
    __connection = None
    __lock = Lock()
    __con_string = f'DRIVER={[item for item in pyodbc.drivers()][-1]};{os.environ["SQLAZURECONNSTR_WWIF"]}'

    def __new__(cls):
        if ConnectionManager.__instance is None:
            ConnectionManager.__instance = object.__new__(cls)        
        return ConnectionManager.__instance       
    
    def __getConnection(self):
        if (self.__connection == None):
            application_name = ";APP={0}".format(socket.gethostname())
            self.__connection = pyodbc.connect(ConnectionManager.__con_string + application_name)                  
        
        return self.__connection

    def __removeConnection(self):
        self.__connection = None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(pyodbc.OperationalError), after=after_log(app.logger, logging.DEBUG))
    def executeQueryJSON(self, procedure, payload=None):
        result = {}  
        try:
            conn = self.__getConnection()

            cursor = conn.cursor()
            
            if payload:
                cursor.execute(f"EXEC {procedure} ?", json.dumps(payload))
            else:
                cursor.execute(f"EXEC {procedure}")

            result = cursor.fetchone()

            if result:
                result = json.loads(result[0])                           
            else:
                result = {}

            cursor.commit()    
        except pyodbc.OperationalError as e:            
            app.logger.error(f"{e.args[1]}")
            if e.args[0] == "08S01":
                # If there is a "Communication Link Failure" error, 
                # then connection must be removed
                # as it will be in an invalid state
                self.__removeConnection() 
                raise                        
        finally:
            cursor.close()
                         
        return result

class Queryable(Resource):
    """
    Full credit: https://github.com/Azure-Samples/azure-sql-db-python-rest-api/blob/master/app.py
    """
    def executeQueryJson(self, verb, payload=None):
        result = {}  
        procedure = f"dbo.{verb}"
        result = ConnectionManager().executeQueryJSON(procedure, payload)
        return result

# ------------------------------------------------------------------------------------------------ #
# Car Price API
# ------------------------------------------------------------------------------------------------ #

class CarPriceModelReview(Queryable):
    def get(self):
        # Query the car inventories with truth
        cars_json = self.executeQueryJson("get_car_inventories_with_truth")

        json_result = car_price_explainer.review_model(cars_json)

        return json_result, 200

class CarPriceSinglePost(Queryable):
    def convert_json_to_dict(self, json_data):
        car_json = {
            'aspiration': json_data['aspiration'],
            'assembled': json_data['assembled'],
            'price': json_data['price'],
            'brand': json_data['brand'],
            'colour': json_data['colour'],
            'direct_injection': json_data['direct_injection'],
            'doors': json_data['doors'],
            'engine_cc': json_data['engine_cc'],
            'fuel_type': json_data['fuel_type'],
            'height_mm': json_data['height_mm'],
            'length_mm': json_data['length_mm'],
            'manufacture_year': json_data['manufacture_year'],
            'mileage': json_data['mileage'],
            'peak_power_hp': json_data['peak_power_hp'],
            'peak_torque_nm': json_data['peak_torque_nm'],
            'seat_capacity': json_data['seat_capacity'],
            'steering_type': json_data['steering_type'],
            'transmission': json_data['transmission'],
            'wheel_base_mm': json_data['wheel_base_mm'],
            'width_mm': json_data['width_mm'],
        }
        return car_json

class CarPricePredictionReview(CarPriceSinglePost):
    def post(self):

        json_data = request.get_json(force=True)

        car_json = self.convert_json_to_dict(json_data)

        json_result = car_price_explainer.review_individual_prediction(car_json)

        return json_result, 200

class CarPriceModelLossReview(CarPriceSinglePost):
    def post(self):

        json_data = request.get_json(force=True)

        car_json = self.convert_json_to_dict(json_data)

        json_result = car_price_explainer.review_individual_model_loss(car_json)

        return json_result, 200

class CarPricePerformanceEval(Queryable):
    def get(self):
        # Query the car inventories with truth
        cars_json = self.executeQueryJson("get_car_inventories_with_truth")

        json_result = car_price_explainer.evaluate_performance(cars_json)

        return json_result, 200

class CarPriceDriftMonitorNoTruth(Queryable):
    def get(self):
        # Query the car inventories without truth
        cars_json = self.executeQueryJson("get_car_inventories_no_truth")

        json_result = car_price_drift_detector.detect_drift(cars_json)

        return json_result, 200

class CarPriceDriftMonitorTruth(Queryable):
    def get(self):
        # Query the car inventories with truth
        cars_json = self.executeQueryJson("get_car_inventories_with_truth")

        json_result = car_price_explainer.detect_drift(cars_json)

        return json_result, 200

class CarPriceUpdateModelAndExplainer(CarPriceSinglePost):
    def post(self):
        json_data = request.get_json(force=True)
        car_json = self.convert_json_to_dict(json_data)

        # Train the adaptive random forest classifier on the lead information in the other web service
        response_ = requests.post('http://river-web-service:80/car/training', json = car_json)
        response_.raise_for_status()

        # Get the JSON containing the extracted tree weights from adaptive random forest classifier
        cp_arf_json = response_.json()
        cp_arf_dict = deserialize_arf(cp_arf_json)

        # Update the lead scoring Tree SHAP explainer and Tree SHAP loss explainer 
        car_price_explainer.update_explainers(cp_arf_dict)
        return 'Successfully updated the car price model and its explainer', 200

api.add_resource(CarPriceModelReview,      '/car/global/review/model')
api.add_resource(CarPricePerformanceEval,  '/car/global/review/performance')
api.add_resource(CarPricePredictionReview, '/car/local/review/prediction')
api.add_resource(CarPriceModelLossReview,  '/car/local/review/model_loss')
api.add_resource(CarPriceUpdateModelAndExplainer,  '/car/update/model')
api.add_resource(CarPriceDriftMonitorTruth,  '/car/global/review/drift/truth')
api.add_resource(CarPriceDriftMonitorNoTruth,  '/car/global/review/drift/no_truth')

# ------------------------------------------------------------------------------------------------ #
# Lead Scoring API
# ------------------------------------------------------------------------------------------------ #

class LeadScoringModelReview(Queryable):
    def get(self):
        # Query the lead information with truth
        leads_json = self.executeQueryJson("get_lead_info_with_truth")

        json_result = lead_scoring_explainer.review_model(leads_json)

        return json_result, 200

class LeadScoringSinglePost(Queryable):
    def convert_json_to_dict(self, json_data):
        lead_json = {
            'dont_email': json_data['dont_email'],
            'dont_call': json_data['dont_call'],
            'occupation': json_data['occupation'],
            'received_free_copy': json_data['received_free_copy'],
            'avg_page_view_per_visit': json_data['avg_page_view_per_visit'],
            'total_site_visit': json_data['total_site_visit'],
            'total_time_spend_on_site': json_data['total_time_spend_on_site'],
            'converted': json_data['converted']
        }
        return lead_json

class LeadScoringPredictionReview(LeadScoringSinglePost):
    def post(self):

        json_data = request.get_json(force=True)

        lead_json = self.convert_json_to_dict(json_data)

        json_result = lead_scoring_explainer.review_individual_prediction(lead_json)

        return json_result, 200

class LeadScoringModelLossReview(LeadScoringSinglePost):
    def post(self):

        json_data = request.get_json(force=True)

        lead_json = self.convert_json_to_dict(json_data)

        json_result = lead_scoring_explainer.review_individual_model_loss(lead_json)

        return json_result, 200

class LeadScoringPerformanceEval(Queryable):
    def get(self):
        # Query the lead information with truth
        leads_json = self.executeQueryJson("get_lead_info_with_truth")

        json_result = lead_scoring_explainer.evaluate_performance(leads_json)

        return json_result, 200

class LeadScoringDriftMonitorNoTruth(Queryable):
    def get(self):
        # Query the lead information without truth
        leads_json = self.executeQueryJson("get_lead_info_no_truth")

        json_result = lead_scoring_drift_detector.detect_drift(leads_json)

        return json_result, 200

class LeadScoringDriftMonitorTruth(Queryable):
    def get(self):
        # Query the lead information with truth
        leads_json = self.executeQueryJson("get_lead_info_with_truth")

        json_result = lead_scoring_explainer.detect_drift(leads_json)

        return json_result, 200

class LeadScoringUpdateModelAndExplainer(LeadScoringSinglePost):
    def post(self):
        json_data = request.get_json(force=True)
        lead_json = self.convert_json_to_dict(json_data)

        # Train the adaptive random forest classifier on the lead information in the other web service
        response_ = requests.post('http://river-web-service:80/lead/training', json = lead_json)
        response_.raise_for_status()

        # Get the JSON containing the extracted tree weights from adaptive random forest classifier
        ls_arf_json = response_.json()
        ls_arf_dict = deserialize_arf(ls_arf_json)

        # Update the lead scoring Tree SHAP explainer and Tree SHAP loss explainer 
        lead_scoring_explainer.update_explainers(ls_arf_dict)
        return 'Successfully updated the lead scoring model and its explainer', 200

api.add_resource(LeadScoringModelReview,      '/lead/global/review/model')
api.add_resource(LeadScoringPerformanceEval,  '/lead/global/review/performance')
api.add_resource(LeadScoringPredictionReview, '/lead/local/review/prediction')
api.add_resource(LeadScoringModelLossReview,  '/lead/local/review/model_loss')
api.add_resource(LeadScoringUpdateModelAndExplainer,  '/lead/update/model')
api.add_resource(LeadScoringDriftMonitorTruth,  '/lead/global/review/drift/truth')
api.add_resource(LeadScoringDriftMonitorNoTruth,  '/lead/global/review/drift/no_truth')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)