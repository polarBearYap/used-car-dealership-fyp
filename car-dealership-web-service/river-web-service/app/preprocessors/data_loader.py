import sys
import pickle

def load_car_price_data_pp(data_pp_path):
    # Fix the pickle import error: https://stackoverflow.com/a/70504686
    sys.path.append('preprocessors')

    with open(data_pp_path, 'rb') as f:
        cp_data_pp = pickle.load(f)

    return cp_data_pp

def load_lead_scoring_data_pp(data_pp_path):
    # Fix the pickle import error: https://stackoverflow.com/a/70504686
    sys.path.append('preprocessors')

    with open(data_pp_path, 'rb') as f:
        ls_data_pp = pickle.load(f)

    return ls_data_pp