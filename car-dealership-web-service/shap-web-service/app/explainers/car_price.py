import json
import numpy as np
import pandas as pd
import shap
from utilities.explainer_visualization import \
plot_model_loss_bar_plot_plotly, plot_beeswarm_plot_plotly, plot_feature_importance_bar_plot_plotly, \
plot_local_bar_plot_plotly, get_shap_monitoring_plot_data, plot_shap_monitoring_plot_plotly
from utilities.general_utils import deserialize_arf, format_car_specs

class CarPriceExplainer:

    # Constant
    FEATURE_LABEL_NAMES = {
        'manufacture_year': 'Manufacture Year',
        'mileage': 'Mileage',
        'length_mm': 'Length (mm)',
        'engine_cc': 'Engine (CC)',
        'aspiration': 'Aspiration',
        'wheel_base_mm': 'Wheel Base (mm)',
        'width_mm': 'Width (mm)',
        'direct_injection': 'Direct Injection',
        'seat_capacity': 'Seat Capacity',
        'peak_power_hp': 'Peak Power Hp',
        'fuel_type': 'Fuel Type',
        'steering_type': 'Steering Type',
        'assembled': 'Assembled',
        'height_mm': 'Height (mm)',
        'peak_torque_nm': 'Peak Torque (nm)',
        'doors': 'Doors',
        'brand': 'Brand',
        'colour': 'Color',
        'model': 'Model',
        'transmission': 'Transmission'
    }
    # Only 600 samples in the train set are used as the validation set
    MAX_TRAIN_SAMPLES = 600

    def __init__(self, cp_data_pp, cp_arf_dict, cp_X_train, cp_y_train, cp_X_train_subsample, cp_X_test_truth_av_subsample) -> None:
        self.cp_data_pp = cp_data_pp
        self.cp_X_train = cp_X_train
        self.cp_y_train = cp_y_train
        self.cp_X_train_subsample = cp_X_train_subsample
        self.cp_X_test_truth_av_subsample = cp_X_test_truth_av_subsample
        self.update_explainers(cp_arf_dict)
        self.cp_shap_loss_values_train_sub = self.cp_tree_loss_explainer_train.shap_values(
            cp_X_train[:self.MAX_TRAIN_SAMPLES], 
            cp_y_train[:self.MAX_TRAIN_SAMPLES]
        )

    def update_explainers(self, cp_arf_dict):
        """
        Initialize tree SHAP explainer and tree SHAP loss explainer.

        Parameters
        ----------
        cp_arf_dict: dict
            The dictionary containing the extracted tree weights from adaptive random forest regressor.
        """

        # Initialize cp_tree_explainer
        self.cp_tree_explainer = shap.TreeExplainer(
            model = cp_arf_dict, 
            feature_perturbation = 'interventional', 
            data = self.cp_X_test_truth_av_subsample
        )

        # Initialize cp_tree_loss_explainer
        self.cp_tree_loss_explainer = shap.TreeExplainer(
            model = cp_arf_dict, 
            feature_perturbation = 'interventional', 
            model_output = 'log_loss', 
            data = self.cp_X_test_truth_av_subsample
        )

        self.cp_tree_loss_explainer_train = shap.TreeExplainer(
            model = cp_arf_dict, 
            feature_perturbation = 'interventional', 
            model_output = 'log_loss',
            data = self.cp_X_train_subsample)

    def review_model(self, cars_dict):
        """
        Public method of reviewing the car price regressor.

        Parameters
        ----------
        cars_dict: list of dict
            List containing the car dictionary records without truth. 
            The `cars_dict` is queried from the database.
        
        json_result: dict
            The dictionary containing the beeswarm plot and the feature importance bar plot.
        """
        car_specifications = pd.DataFrame(cars_dict)

        # Preprocess the test set
        car_specifications = format_car_specs(car_specifications)
        car_specifications = self.cp_data_pp.preprocess(car_specifications)

        # Calculate SHAP values
        cp_shap_values = self.cp_tree_explainer.shap_values(car_specifications)

        # Constsruct the beeswarm plot using Plotly
        beeswarm_fig = plot_beeswarm_plot_plotly(car_specifications, cp_shap_values, self.FEATURE_LABEL_NAMES)

        # Construct the feature importance bar plot using Plotly
        fi_bar_fig = plot_feature_importance_bar_plot_plotly(car_specifications, cp_shap_values, self.FEATURE_LABEL_NAMES)

        # Export to HTML
        beeswarm_fig_html = beeswarm_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )

        fi_bar_fig_html = fi_bar_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )
        
        json_result = {
            'beeswarm': beeswarm_fig_html,
            'bar': fi_bar_fig_html
        }

        return json_result

    def review_individual_prediction(self, car_dict):
        """
        Public method of reviewing the individual car price prediction.

        Parameters
        ----------
        car_dict: dict
            Dictionary containing a single car records without truth. 
            The `car_dict` is send by the client in the web application.
        
        json_result: dict
            The dictionary containing the SHAP bar plot.
        """
        car_specification = pd.DataFrame([car_dict])

        # Preprocess the test set
        car_specification = format_car_specs(car_specification)
        car_specification = self.cp_data_pp.preprocess(car_specification)

        # Calculate SHAP values
        cp_shap_values = self.cp_tree_explainer.shap_values(car_specification)

        # Construct the SHAP bar plot using Plotly
        shap_bar_fig = plot_local_bar_plot_plotly(
            shap_values = cp_shap_values[0],
            features = car_specification.values[0, :],
            feature_names = car_specification.columns.tolist(), 
            feature_label_names = self.FEATURE_LABEL_NAMES
        )

        # Export to HTML
        shap_bar_fig_html = shap_bar_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )

        json_result = {
            'shap': shap_bar_fig_html
        }

        return json_result

    def review_individual_model_loss(self, car_dict):
        """
        Public method of reviewing the individual model loss.

        Parameters
        ----------
        car_dict: dict
            Dictionary containing a single car records with truth. 
            The `car_dict` is send by the client in the web application.
        
        json_result: dict
            The dictionary containing the SHAP bar plot and SHAP loss bar plot.
        """
        car_specification = pd.DataFrame([car_dict])

        price = car_specification.loc[0, 'price']

        # Preprocess the test set
        car_specification = format_car_specs(car_specification)
        car_specification = self.cp_data_pp.preprocess(car_specification)

        # Calculate SHAP values
        cp_shap_values = self.cp_tree_explainer.shap_values(car_specification)

        # Calculate SHAP loss values
        cp_shap_loss_values = self.cp_tree_loss_explainer.shap_values(car_specification, [price])

        # Construct the SHAP bar plot using Plotly
        shap_bar_fig = plot_local_bar_plot_plotly(
            shap_values = cp_shap_values[0],
            features = car_specification.values[0, :],
            feature_names = car_specification.columns.tolist(), 
            feature_label_names = self.FEATURE_LABEL_NAMES
        )

        # Calculate the SHAP loss values
        shap_loss_bar_fig = plot_local_bar_plot_plotly(
            shap_values = cp_shap_loss_values[0],
            features = car_specification.values[0, :],
            feature_names = car_specification.columns.tolist(), 
            feature_label_names = self.FEATURE_LABEL_NAMES
        )

        # Export to HTML
        shap_bar_fig_html = shap_bar_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )

        shap_loss_bar_fig_html = shap_loss_bar_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )

        json_result = {
            'shap': shap_bar_fig_html,
            'shap_loss': shap_loss_bar_fig_html
        }

        return json_result

    def evaluate_performance(self, cars_dict):
        """
        Public method of evaluating the performance of car price regressor.

        Parameters
        ----------
        cars_dict: list of dict
            List containing the car dictionary records with truth. 
            The `cars_dict` is queried from the database.
        
        json_result: dict
            The dictionary containing the beeswarm plot and the feature importance bar plot.
        """
        car_specifications = pd.DataFrame(cars_dict)

        car_prices = car_specifications['price'].values

        # Preprocess the test set
        car_specifications = format_car_specs(car_specifications)
        car_specifications = self.cp_data_pp.preprocess(car_specifications)

        # Calculate SHAP loss values
        cp_shap_loss_values = self.cp_tree_loss_explainer.shap_values(car_specifications, car_prices)

        # Construct the positive SHAP loss bar plot
        pos_shap_loss_bar_fig = plot_model_loss_bar_plot_plotly(
            features = car_specifications, 
            shap_values = cp_shap_loss_values, 
            feature_label_names = self.FEATURE_LABEL_NAMES, 
            shap_loss_type = 'positive'
        )

        # Construct the negative SHAP loss bar plot
        neg_shap_loss_bar_fig = plot_model_loss_bar_plot_plotly(
            features = car_specifications, 
            shap_values = cp_shap_loss_values, 
            feature_label_names = self.FEATURE_LABEL_NAMES, 
            shap_loss_type = 'negative'
        )

        # Export to HTML
        pos_shap_loss_bar_fig_html = pos_shap_loss_bar_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )

        neg_shap_loss_bar_fig_html = neg_shap_loss_bar_fig.to_html(
            include_plotlyjs='cdn', 
            default_width='100%', 
            default_height='100%', 
            validate=True, 
            full_html=False, 
        )

        json_result = {
            'pos': pos_shap_loss_bar_fig_html,
            'neg': neg_shap_loss_bar_fig_html
        }

        return json_result

    def detect_drift(self, cars_dict):
        """
        Public method of monitoring the drift using SHAP loss monitoring function.

        Parameters
        ----------
        cars_dict: list of dict
            List containing the car dictionary records with truth. 
            The `cars_dict` is queried from the database.
        
        json_result: dict
            The dictionary containing the SHAP monitoring loss plots and the alarms.
        """

        car_specifications = pd.DataFrame(cars_dict)

        car_prices = car_specifications['price'].values

        # Preprocess the test set
        car_specifications = format_car_specs(car_specifications)
        car_specifications = self.cp_data_pp.preprocess(car_specifications)
        # Calculate the SHAP loss value
        cp_shap_loss_values_test = self.cp_tree_loss_explainer.shap_values(car_specifications, car_prices)

        # To reduce the computation time, only the top 20 features with highest positive SHAP loss values are considered
        shap_loss_values = pd.DataFrame(cp_shap_loss_values_test, columns = self.cp_X_train.columns)
        shap_loss_values_summed = shap_loss_values.apply(lambda col: col[col >= 0.0].sum(), axis=0)
        top_20_inaccurate_features = shap_loss_values_summed.sort_values(ascending=False).index.tolist()[:20]

        problematic_cols = []

        # Get features that trigger alarms
        for idx, feature_name in enumerate(top_20_inaccurate_features):
            # Call the SHAP monitoring function
            _, alarm_info = get_shap_monitoring_plot_data(
                feature_name = feature_name,
                shap_values_list = [
                    self.cp_shap_loss_values_train_sub,
                    cp_shap_loss_values_test
                ],
                features_list = [
                    self.cp_X_train[:self.MAX_TRAIN_SAMPLES], 
                    car_specifications
                ], 
                feature_names = self.cp_X_train.columns.tolist(), 
                increment = 250
            )
            # If the alarm is triggered
            if 'p-value' in alarm_info:
                # Save the problematic columns to construct the plot later
                problematic_cols.append(feature_name)

        shap_loss_monitoring_fig_htmls = []
        alarms = []

        for feature_name in problematic_cols:
            # Get the SHAP monitoring loss plot and the alarm information
            fig, alarm_info = plot_shap_monitoring_plot_plotly(
                feature_name = feature_name,
                shap_values_list = [
                    self.cp_shap_loss_values_train_sub, 
                    cp_shap_loss_values_test
                ],
                features_list = [
                    self.cp_X_train[:self.MAX_TRAIN_SAMPLES], 
                    car_specifications
                ], 
                feature_names = self.cp_X_train.columns.tolist(), 
                feature_label_names = self.FEATURE_LABEL_NAMES
            )
            # Export to HTML
            fig_html = fig.to_html(
                include_plotlyjs='cdn', 
                default_width='100%', 
                default_height='100%', 
                validate=True, 
                full_html=False, 
            )
            # Convert the numpy data type to basic data type before coverting to JSON
            for key in alarm_info.keys():
                if type(alarm_info[key]) == np.int64:
                    alarm_info[key] = int(alarm_info[key])
                elif type(alarm_info[key]) == np.float64:
                    alarm_info[key] = float(alarm_info[key])
            alarms.append(alarm_info)
            shap_loss_monitoring_fig_htmls.append(fig_html)

        json_result = {
            'alarms': alarms,
            'figs': shap_loss_monitoring_fig_htmls
        }

        return json_result
