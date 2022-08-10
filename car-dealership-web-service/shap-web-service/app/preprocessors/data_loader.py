import pandas as pd
import pickle
import sys

def load_car_price_data(data_pp_path):
    # Fix the pickle import error: https://stackoverflow.com/a/70504686
    sys.path.append('preprocessors')

    with open(data_pp_path, 'rb') as f:
        cp_data_pp = pickle.load(f)

    # Load car price test subsamples to initialize SHAP loss explainer for test set
    cp_X_test_truth_av_subsample = pd.read_csv('data/cp_X_test_truth_av_subsample.csv')

    return cp_data_pp, cp_X_test_truth_av_subsample

def load_lead_scoring_data(data_pp_path):
    # Fix the pickle import error: https://stackoverflow.com/a/70504686
    sys.path.append('preprocessors')

    with open(data_pp_path, 'rb') as f:
        ls_data_pp = pickle.load(f)

    # Load lead scoring subsamples to initialize SHAP loss explainer for test set
    ls_X_test_truth_av_subsample = pd.read_csv('data/ls_X_test_truth_av_subsample.csv')
    
    return ls_data_pp, ls_X_test_truth_av_subsample