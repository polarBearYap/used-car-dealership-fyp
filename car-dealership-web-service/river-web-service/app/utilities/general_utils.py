import numpy as np
import pandas as pd

def serialize_arf(arf_dict):
    """
    Make the dictionary convertible to JSON by converting all numpy arrays to Python list 
    and converting type to str.

    Parameters
    ----------
    arf_dict: dict
        The dictionary containing the extracted tree weights.
        
    Returns
    -------
    arf_dict_serializable : dict
        The dictionary containing the extracted tree weights that can be converted to JSON.
    """
    arf_dict_serializable = {
        'internal_dtype': str(arf_dict['internal_dtype']),
        'input_dtype': str(arf_dict['input_dtype']),
        'objective'     : arf_dict['objective'],
        'tree_output'   : arf_dict['tree_output'],
        'base_offset': arf_dict['base_offset'],
        'trees': []
    }

    for tree in arf_dict['trees']:
        ht_dict_serializable = {}
        for weight_type, arr in tree.items():
            ht_dict_serializable[weight_type] = arr.tolist()
        arf_dict_serializable['trees'].append(ht_dict_serializable)

    return arf_dict_serializable

def deserialize_arf(arf_dict_serializable):
    """
    Convert all Python lists back to numpy arrays and convert str back to type. 

    Parameters
    ----------
    arf_dict: dict
        The dictionary containing the extracted tree weights.
        
    Returns
    -------
    arf_dict_serializable : dict
        The dictionary containing the extracted tree weights that can be converted to JSON.
    """
    arf_dict = {
        'objective'  : arf_dict_serializable['objective'],
        'tree_output': arf_dict_serializable['tree_output'],
        'base_offset': int(arf_dict_serializable['base_offset']),
        'trees': []
    }

    for key in ['internal_dtype', 'input_dtype']:
        if 'numpy.float64' in arf_dict_serializable[key]:
            arf_dict[key] = np.float64
        else:
            arf_dict[key] = np.float32

    for tree in arf_dict_serializable['trees']:
        ht_dict = {}
        for weight_type, arr in tree.items():
            ht_dict[weight_type] = np.array(arr, dtype=arf_dict['internal_dtype'])
        arf_dict['trees'].append(ht_dict)

    return arf_dict

def format_car_specs(car_specifications):
    """
    Convert some categorical values in the car specifications
    so that one-hot encoder can preprocess.
    
    Parameters
    ----------
    car_specifications: pandas.core.frame.DataFrame
        The car specifications.

    Returns
    -------
    car_specifications: pandas.core.frame.DataFrame
        The car specifications that the one-hot encoder can preprocess.
    """
    
    assert type(car_specifications) == pd.core.frame.DataFrame, \
    'car_specifications must be a dataframe'
    
    # Set the null value in colour column to ' -'
    color_null_indexes = car_specifications['colour'] == 'null'
    car_specifications.loc[color_null_indexes, 'colour'] = ' -'
    
    return car_specifications

def format_lead_infos(lead_information):
    """
    Convert some categorical values in the lead records
    so that one-hot encoder can preprocess.
    
    Parameters
    ----------
    lead_information: pandas.core.frame.DataFrame
        Dataframe containing lead information.

    Returns
    -------
    lead_information: pandas.core.frame.DataFrame
        The lead information that the one-hot encoder can preprocess.
    """
    
    assert type(lead_information) == pd.core.frame.DataFrame, \
    'lead_information must be a dataframe'
    
    # Set the value in occupation column from to 'Currently Not Employed' to 'Unemployed'
    indexes = lead_information['occupation'] == 'Currently Not Employed'
    lead_information.loc[indexes, 'occupation'] = 'Unemployed'

    # Set the value in occupation column from to 'Business person' to 'Businessman'
    indexes = lead_information['occupation'] == 'Business person'
    lead_information.loc[indexes, 'occupation'] = 'Businessman'
    
    return lead_information
