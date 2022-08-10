# Author: Yap Jheng Khin

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor():
    """
    A custom class used to preprocess data.


    Attributes
    ----------
    X : DataFrame
        Feature values used to train the data processing pipeline.
    num_attrs: List of str
        List containing numerical attributes.
    cat_attrs: List of str
        List containing categorical attributes.

    Methods
    -------
    _init_preprocess(X, num_attrs, cat_attrs)
        Initialize the one hot encoder.
    preprocess_X(cur_X)
        Preprocess feature values.
    """
    
    def __init__(self, X, num_attrs, cat_attrs):
        """
        Parameters
        ----------
        X : DataFrame
            Feature values used to train the data processing pipeline.
        num_attrs: List of str
            List containing numerical attributes.
        cat_attrs: List of str
            List containing categorical attributes.
        """
        self.X = X
        self.num_attrs = num_attrs
        self.cat_attrs = cat_attrs
        self._init_preprocess(X, num_attrs, cat_attrs)
    
    @property
    def features(self):
        """Get the nominal attribute values.
        
        Returns
        ----------
        features: list of str
            List containing names for numerical features and nominal features.
        """
        return list(self.num_attrs) + list(self.nominal_attrs)

    def _init_preprocess(self, X, num_attrs, cat_attrs):
        """Initialize the one hot encoder.
        
        Parameters
        ----------
        X : DataFrame
            Feature values.
        num_attrs: List of str
            List containing numerical attributes.
        cat_attrs: List of str
            List containing categorical attributes.
        """
        self.nominal_attrs = []
        if (len(cat_attrs) > 0):
            self.one_hot_enc = OneHotEncoder(
                sparse=False, 
                drop = 'if_binary', 
                handle_unknown='ignore'
            )
            self.one_hot_enc.fit(self.X[self.cat_attrs])
            self.nominal_attrs = list(self.one_hot_enc.get_feature_names_out(cat_attrs))
        
    def preprocess(self, cur_X):
        """Preprocess feature values.
        
        Parameters
        ----------
        X : DataFrame
            Feature values to be preprocessed.
        
        Returns
        ----------
        X_pp: DataFrame
            Preprocessed features.
        """
        X_pp = cur_X[self.num_attrs].values

        if (len(self.nominal_attrs) > 0):
            X_pp_cat = self.one_hot_enc.transform(cur_X[self.cat_attrs])
            X_pp = np.hstack((X_pp, X_pp_cat))

        return pd.DataFrame(X_pp, columns = self.features)