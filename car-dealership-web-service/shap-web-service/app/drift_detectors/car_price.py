import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from utilities.feature_dist_monitoring import \
    cal_chi2, cal_psi, determine_cal_methods, display_chi2, display_psi, \
    get_category_count, get_histogram_count, plot_psi_graph
from utilities.general_utils import format_car_specs

class CarPriceDriftDetector:

    RANDOM_SEED = 2022
    CAR_PRICE_BIN_THRESHOLD = 10

    def __init__(self, cp_X_train, cp_data_pp) -> None:
        self.cp_X_train = cp_X_train
        self.cp_data_pp = cp_data_pp
        self._determine_calculation_type()

    def _determine_calculation_type(self):
        """
        Determine if each X feature is suitable to be calculated using PSI, 
        chi-squared goodness of fit test, or neither of the methods.
        """

        cp_X_train_ = self.cp_X_train.copy()
        # `doors` and `seat_capacity` are considered as ordinal features, 
        # thus they are considered as categorical features
        cp_X_train_ = cp_X_train_.astype({'doors': 'object', 'seat_capacity': 'object'})

        self.cp_cols_calculated_by_psi, self.cp_cols_calculated_by_chi, _ = \
        determine_cal_methods(cp_X_train_, self.CAR_PRICE_BIN_THRESHOLD)

    def detect_drift(self, cars_dict):
        """
        Public method of detecting drift on car records without truth.

        Parameters
        ----------
        cars_dict: list of dict
            List containing the car dictionary records without truth. 
            The `cars_dict` is queried from the database.

        Returns
        -------
        json_result: dict
            The dictionary containing the PSI plots, PSI tables, and chi-squared tables in HTML format.
        """
        car_specifications = pd.DataFrame(cars_dict)

        # Preprocess the test set
        car_specifications = format_car_specs(car_specifications)
        car_specifications = self.cp_data_pp.preprocess(car_specifications)

        # Deliberately drift the data for system testing.
        cp_X_test_drifted = self._drift_data(car_specifications)
        
        # Use PSI to detect drift
        psi_html_plots, psi_html_tables = self._check_drift_using_psi(cp_X_test_drifted)

        # Use chi-squared goodness of fit test to detect drift
        chi2_html_tables = self._check_drift_using_chi2(cp_X_test_drifted)

        json_result = {
            'psi': {
                'plots': psi_html_plots,
                'tables': psi_html_tables
            },
            'chi2': {
                'tables': chi2_html_tables
            }
        }

        return json_result

    def _check_drift_using_chi2(self, cp_X_test_drifted):
        """
        For each feature that can be calculated using chi-squared goodness of fit test,
        check if the distribution difference between expected and observed values are statistically significant 

        Parameters
        ----------
        cp_X_test_drifted: pandas.core.frame.DataFrame
            The drifted car specifications.

        Returns
        -------
        chi2_html_tables: list
            The list containing the formatted Chi-squared tables in HTML format.
        """

        # For each feature that can be calculated using chi-squared goodness of fit test,
        # check if the distribution difference between expected and observed values are statistically significant 
        df = pd.DataFrame([], columns=['feature_name', 'chi2_value', 'p_value'])

        for idx, feature_name in enumerate(self.cp_cols_calculated_by_chi):
            exp_cat_counts, obs_cat_counts = get_category_count(self.cp_X_train, 
                                                                cp_X_test_drifted, 
                                                                feature_name)
            chi2_table = cal_chi2(exp_cat_counts, obs_cat_counts, feature_name)
            chi2_val = chi2_table["Chi2 value"].sum()
            p_val = scipy_stats.chi2.sf(chi2_val, len(exp_cat_counts) - 1)
            df.loc[idx, :] = [feature_name, round(chi2_val, 4), round(p_val, 4)]

        # Get the drifted features
        drift_df = df[df['p_value'] <= 0.05]

        drifted_chi2_cols = drift_df['feature_name'].tolist()
        chi2_html_tables = []

        for feature_name in drifted_chi2_cols:
            # Get expected and observed counts
            exp_cat_counts, obs_cat_counts = get_category_count(
                self.cp_X_train, 
                cp_X_test_drifted, 
                feature_name
            )
            # Calculate the chi-squared goodness of fit test and get its table
            chi2_table = cal_chi2(exp_cat_counts, obs_cat_counts, feature_name, display=False)
            # Format the chi2 table
            chi2_table_display = display_chi2(chi2_table, show_sum=True)
            # Drop columns that are not important to display
            chi2_table_display = chi2_table_display.drop(columns='Squared Diff count', axis=1)
            # Export the table to HTML
            chi2_table_html = chi2_table_display.reset_index()
            chi2_table_html = chi2_table_html.rename(columns={'index': 'Category'})
            chi2_table_html.loc[len(chi2_table_html)-1, 'Category'] = f'<b>Total</b>'
            chi2_table_html = chi2_table_html.to_html(header=True, index=False, 
                                                    notebook=False, escape=False)
            chi2_html_tables.append(chi2_table_html)
        
        return chi2_html_tables

    def _check_drift_using_psi(self, cp_X_test_drifted):
        """
        For each feature that can be calculated using PSI, check if the distribution 
        difference between expected and observed values are statistically significant.

        Parameters
        ----------
        cp_X_test_drifted: pandas.core.frame.DataFrame
            The drifted car specifications.

        Returns
        -------
        psi_html_plots: list
            The list containing the PSI plots in HTML format.
        psi_html_tables: list
            The list containing the formatted PSI tables in HTML format.
        """

        df = pd.DataFrame([], columns=['feature_name', 'psi_value'])

        for idx, feature_name in enumerate(self.cp_cols_calculated_by_psi):
            expected_freq, observed_freq, bin_edges = \
            get_histogram_count(self.cp_X_train, cp_X_test_drifted, feature_name, self.CAR_PRICE_BIN_THRESHOLD)
            psi_table = cal_psi(expected_freq, observed_freq)
            df.loc[idx, :] = [feature_name, round(psi_table["psi_val"].sum(), 4)]

        potential_drift_df = df[(df['psi_value'] >= 0.1) & (df['psi_value'] <= 0.25)]
        drift_df = df[df['psi_value'] > 0.25]

        # Get the features that may have or have drifted
        drifted_psi_cols = drift_df['feature_name'].tolist() + potential_drift_df['feature_name'].tolist()
        psi_html_tables = []
        psi_html_plots = []

        for feature_name in drifted_psi_cols:
            # Get the expected frequency and observed frequency
            expected_freq, observed_freq, bin_edges = \
            get_histogram_count(self.cp_X_train, cp_X_test_drifted, feature_name, self.CAR_PRICE_BIN_THRESHOLD)
            # Calculate PSI and get its table
            psi_table = cal_psi(expected_freq, observed_freq)
            # Format the PSI table
            psi_table_display = display_psi(psi_table, bin_edges, show_percent=True, show_sum=True)
            # Drop columns that are not important to display
            psi_table_display = psi_table_display.drop(columns='log_e_diff_relative_freq', axis=1)
            # Export the table to HTML
            psi_table_html = psi_table_display.reset_index()
            psi_table_html = psi_table_html.rename(columns={'index': 'Bin'})
            psi_table_html.loc[len(psi_table_html)-1, 'Bin'] = f'<b>Total</b>'
            html_table = psi_table_html.to_html(header=True, index=False, notebook=False, escape=False)
            psi_html_tables.append(html_table)
            psi_fig = plot_psi_graph(psi_table, bin_edges)
            # Export the plot to HTML
            psi_fig_html = psi_fig.to_html(
                include_plotlyjs='cdn', 
                default_width='100%', 
                default_height='100%', 
                validate=True, 
                full_html=False, 
            )
            psi_html_plots.append(psi_fig_html)

        return psi_html_plots, psi_html_tables
    
    def _drift_data(self, car_specifications):
        """Deliberately drift the data for system testing.

        Parameters
        ----------
        car_specifications: pandas.core.frame.DataFrame
            Dataframe containing car specifications.
        
        Returns
        -------
        cp_X_test_drifted: pandas.core.frame.DataFrame
            Dataframe containing drifted car specifications.
        """
        # Deliberately apply drift to the test set for system testing
        cp_X_test_drifted = car_specifications.copy()

        # Drift the `transmission_Manual`
        # Increase the positive samples `transmission_Manual` by 5%
        # Assuming that more customers less prefer second-used cars with manual transmission overtime
        indexes = cp_X_test_drifted['transmission_Manual'] == 0
        rng = np.random.default_rng(self.RANDOM_SEED)
        cp_X_test_drifted.loc[indexes, 'transmission_Manual'] = \
        rng.choice([0, 1], p=[0.95, 0.05], size=sum(indexes))

        # Simulate data error for numerical features 
        # Device the mileage by 1.609
        # Assuming that the users accidentally enter miles instead of kilometer
        cp_X_test_drifted['mileage'] /= 1.609

        # Simulate gradual feature drift
        # Gradually drift the manufacture year in 3 stages
        # Each stages will increase the year offsets by 1
        # Assuming that the car price model has already been used for years
        # Car inventory information with higher `manufacture_year` is observed more frequently 
        num_samples = len(cp_X_test_drifted)
        period = 3
        data_gap = round(num_samples / period, -1)
        years_coeff = (0, 1)
        rng = np.random.default_rng(self.RANDOM_SEED)

        for i in range(period):
            # Update the offset of manufacture years by 1
            years_coeff = (years_coeff[0]+1, years_coeff[1]+1)
            # Update the start and end index
            start = i*data_gap
            end = (i+1)*data_gap
            cur_sample_count = len(cp_X_test_drifted.loc[start:end])
            years_offset = rng.choice(range(years_coeff[0], years_coeff[1]), size=cur_sample_count)
            cp_X_test_drifted.loc[start:end, 'manufacture_year'] += years_offset

        return cp_X_test_drifted
