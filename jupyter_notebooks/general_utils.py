import numpy as np

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