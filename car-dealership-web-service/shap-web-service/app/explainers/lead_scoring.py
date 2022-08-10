import numpy as np
import pandas as pd
import shap
from utilities.explainer_visualization import \
plot_model_loss_bar_plot_plotly, plot_beeswarm_plot_plotly, plot_feature_importance_bar_plot_plotly, \
plot_local_bar_plot_plotly, get_shap_monitoring_plot_data, plot_shap_monitoring_plot_plotly
from utilities.general_utils import format_lead_infos

class LeadScoringExplainer:

    FEATURE_LABEL_NAMES = {
        'total_site_visit': 'Total Site Visit', 
        'total_time_spend_on_site': 'Total Time Spend On Site',
        'avg_page_view_per_visit': 'Average Page View Per Visit', 
        'dont_email_Yes': 'Don\'t Email_Yes', 
        'occupation_Businessman': 'Occupation_Businessman',
        'occupation_Student': 'Occupation_Student', 
        'occupation_Unemployed': 'Occupation_Unemployed',
        'occupation_Working Professional': 'Occupation_Working Professional', 
        'received_free_copy_Yes': 'Received Free Copy_Yes'
    }
    # Only 1400 samples in the train set are used as the validation set
    MAX_TRAIN_SAMPLES = 1400

    def __init__(self, ls_data_pp, ls_arf_dict, ls_X_train, ls_y_train, ls_X_train_subsample, ls_X_test_truth_av_subsample) -> None:
        self.ls_data_pp = ls_data_pp
        self.ls_X_train = ls_X_train
        self.ls_y_train = ls_y_train
        self.ls_X_train_subsample = ls_X_train_subsample
        self.ls_X_test_truth_av_subsample = ls_X_test_truth_av_subsample
        self.update_explainers(ls_arf_dict)
        self.ls_shap_loss_values_train_sub = self.ls_tree_loss_explainer_train.shap_values(
            ls_X_train[:self.MAX_TRAIN_SAMPLES], 
            ls_y_train[:self.MAX_TRAIN_SAMPLES]
        )

    def update_explainers(self, ls_arf_dict):
        """
        Initialize tree SHAP explainer and tree SHAP loss explainer.

        Parameters
        ----------
        ls_arf_dict: dict
            The dictionary containing the extracted tree weights from adaptive random forest regressor.
        """

        self.ls_tree_explainer = shap.TreeExplainer(
            model = ls_arf_dict, 
            feature_perturbation = 'interventional', 
            data = self.ls_X_test_truth_av_subsample
        )

        self.ls_tree_loss_explainer = shap.TreeExplainer(
            model = ls_arf_dict, 
            feature_perturbation = 'interventional', 
            model_output = 'log_loss',
            data = self.ls_X_test_truth_av_subsample
        )

        self.ls_tree_loss_explainer_train = shap.TreeExplainer(
            model = ls_arf_dict, 
            feature_perturbation = 'interventional', 
            model_output = 'log_loss',
            data = self.ls_X_train_subsample)

    def review_model(self, leads_dict):
        """
        Public method of reviewing the lead price regressor.

        Parameters
        ----------
        leads_dict: list of dict
            List containing the lead dictionary records without truth. 
            The `leads_dict` is queried from the database.

        json_result: dict
            The dictionary containing the beeswarm plot and the feature importance bar plot.
        """
        lead_information = pd.DataFrame(leads_dict)

        # Preprocess the test set
        lead_information = format_lead_infos(lead_information)
        lead_information = self.ls_data_pp.preprocess(lead_information)

        # Calculate SHAP values
        ls_shap_values = self.ls_tree_explainer.shap_values(lead_information)

        # Constsruct the beeswarm plot using Plotly
        beeswarm_fig = plot_beeswarm_plot_plotly(lead_information, ls_shap_values[1], self.FEATURE_LABEL_NAMES)
        # Override figure configuration
        beeswarm_fig.update_traces(hovertemplate = '<i>Feature:</i> %{y}<br><i>Sum of SHAP absolute values</i>: %{x:.4f}<br>')

        # Construct the feature importance bar plot using Plotly
        fi_bar_fig = plot_feature_importance_bar_plot_plotly(lead_information, ls_shap_values[1], self.FEATURE_LABEL_NAMES)
        # Override figure configuration
        fi_bar_fig.update_traces(hovertemplate = '<i>Feature:</i> %{y}<br><i>Sum of SHAP absolute values</i>: %{x:.4f}<br>')

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

    def review_individual_prediction(self, lead_dict):
        """
        Public method of reviewing the individual lead price prediction.

        Parameters
        ----------
        lead_dict: dict
            Dictionary containing a single lead records without truth. 
            The `lead_dict` is send by the client in the web application.
        
        json_result: dict
            The dictionary containing the SHAP bar plot.
        """
        lead_information = pd.DataFrame([lead_dict])

        # Preprocess the test set
        lead_information = format_lead_infos(lead_information)
        lead_information = self.ls_data_pp.preprocess(lead_information)

        # Calculate SHAP values
        ls_shap_values = self.ls_tree_explainer.shap_values(lead_information)

        # Construct the SHAP bar plot using Plotly
        shap_bar_fig = plot_local_bar_plot_plotly(
            shap_values = ls_shap_values[1][0],
            features = lead_information.values[0, :],
            feature_names = lead_information.columns.tolist(), 
            feature_label_names = self.FEATURE_LABEL_NAMES
        )
        # Override figure configuration
        shap_bar_fig.update_traces(
            hovertemplate = '<i>Feature:</i> %{y}<br>'+
            '<i>Feature value</i>: %{customdata:.0f}<br>'+
            '<i>SHAP values</i>: %{x:.4f}<br>',
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

    def review_individual_model_loss(self, lead_dict):
        """
        Public method of reviewing the individual model loss.

        Parameters
        ----------
        lead_dict: dict
            Dictionary containing a single lead records with truth. 
            The `lead_dict` is send by the client in the web application.
        
        json_result: dict
            The dictionary containing the SHAP bar plot and SHAP loss bar plot.
        """
        lead_information = pd.DataFrame([lead_dict])

        lead_status = lead_information.loc[0, 'converted']

        # Preprocess the test set
        lead_information = format_lead_infos(lead_information)
        lead_information = self.ls_data_pp.preprocess(lead_information)

        # Calculate SHAP values
        ls_shap_values = self.ls_tree_explainer.shap_values(lead_information)

        # Calculate SHAP loss values
        ls_shap_loss_values = self.ls_tree_loss_explainer.shap_values(lead_information, [lead_status])

        # Construct the SHAP bar plot using Plotly
        shap_bar_fig = plot_local_bar_plot_plotly(
            shap_values = ls_shap_values[1][0],
            features = lead_information.values[0, :],
            feature_names = lead_information.columns.tolist(), 
            feature_label_names = self.FEATURE_LABEL_NAMES
        )
        # Override figure configuration
        shap_bar_fig.update_traces(
            hovertemplate = '<i>Feature:</i> %{y}<br>'+
            '<i>Feature value</i>: %{customdata:.0f}<br>'+
            '<i>SHAP values</i>: %{x:.4f}<br>',
        )

        # Construct the SHAP loss bar plot using Plotly
        shap_loss_bar_fig = plot_local_bar_plot_plotly(
            shap_values = ls_shap_loss_values[1][0],
            features = lead_information.values[0, :],
            feature_names = lead_information.columns.tolist(), 
            feature_label_names = self.FEATURE_LABEL_NAMES,
            shap_type = 'SHAP loss value'
        )
        # Override figure configuration
        shap_loss_bar_fig.update_traces(
            hovertemplate = '<i>Feature:</i> %{y}<br>'+
            '<i>Feature value</i>: %{customdata:.0f}<br>'+
            '<i>SHAP loss values</i>: %{x:.4f}<br>',
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

    def evaluate_performance(self, leads_dict):
        """
        Public method of evaluating the performance of lead price regressor.

        Parameters
        ----------
        leads_dict: list of dict
            List containing the lead dictionary records with truth. 
            The `leads_dict` is queried from the database.
        
        json_result: dict
            The dictionary containing the negative and positive SHAP loss bar plots.
        """
        lead_information = pd.DataFrame(leads_dict)

        lead_statuses = lead_information['converted'].values

        # Preprocess the test set
        lead_information = format_lead_infos(lead_information)
        lead_information = self.ls_data_pp.preprocess(lead_information)

        # Calculate SHAP loss values
        ls_shap_loss_values = self.ls_tree_loss_explainer.shap_values(lead_information, lead_statuses)

        # Construct the positive SHAP loss bar plot
        pos_shap_loss_bar_fig = plot_model_loss_bar_plot_plotly(
            features = lead_information, 
            shap_values = ls_shap_loss_values[1], 
            feature_label_names = self.FEATURE_LABEL_NAMES, 
            shap_loss_type = 'positive'
        )
        # Override figure configuration
        pos_shap_loss_bar_fig.update_traces(hovertemplate = '<i>Feature:</i> %{x}<br><i>Sum of SHAP loss values</i>: %{y:.4f}<br>')

        # Construct the negative SHAP loss bar plot
        neg_shap_loss_bar_fig = plot_model_loss_bar_plot_plotly(
            features = lead_information, 
            shap_values = ls_shap_loss_values[1], 
            feature_label_names = self.FEATURE_LABEL_NAMES, 
            shap_loss_type = 'negative'
        )
        # Override figure configuration
        neg_shap_loss_bar_fig.update_traces(hovertemplate = '<i>Feature:</i> %{x}<br><i>Sum of SHAP loss values</i>: %{y:.4f}<br>')

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

    def detect_drift(self, leads_dict):
        """
        Public method of evaluating the performance of lead price regressor.

        Parameters
        ----------
        leads_dict: list of dict
            List containing the lead dictionary records with truth. 
            The `leads_dict` is queried from the database.
        
        json_result: dict
            The dictionary containing the SHAP monitoring loss plots and the alarms.
        """
        lead_information = pd.DataFrame(leads_dict)

        lead_statuses = lead_information['converted'].values

        # Preprocess the test set
        lead_information = format_lead_infos(lead_information)
        lead_information = self.ls_data_pp.preprocess(lead_information)
        # Calculate the SHAP loss value
        ls_shap_loss_values_test = self.ls_tree_loss_explainer.shap_values(lead_information, lead_statuses)

        problematic_cols = []

        # Get features that trigger alarms
        for idx, feature_name in enumerate(self.ls_X_train.columns.tolist()):
            # Call the SHAP monitoring function
            _, alarm_info = get_shap_monitoring_plot_data(
                feature_name = feature_name,
                shap_values_list = [
                    self.ls_shap_loss_values_train_sub[1],
                    ls_shap_loss_values_test[1]
                ],
                features_list = [
                    self.ls_X_train[:self.MAX_TRAIN_SAMPLES], 
                    lead_information
                ], 
                feature_names = self.ls_X_train.columns.tolist(), 
                increment = 50
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
                    self.ls_shap_loss_values_train_sub[1], 
                    ls_shap_loss_values_test[1]
                ],
                features_list = [
                    self.ls_X_train[:self.MAX_TRAIN_SAMPLES], 
                    lead_information
                ], 
                feature_names = self.ls_X_train.columns.tolist(), 
                feature_label_names = self.FEATURE_LABEL_NAMES
            )
            # Override figure configuration
            fig.update_traces(
                hovertemplate = \
                f'<i>{self.FEATURE_LABEL_NAMES[feature_name]}</i>: '+
                '%{customdata:.0f}<br>'+
                '<i>Sample index</i>: %{x:.0f}<br>' +
                '<i>SHAP loss value</i>: %{y:.4f}<br>'
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