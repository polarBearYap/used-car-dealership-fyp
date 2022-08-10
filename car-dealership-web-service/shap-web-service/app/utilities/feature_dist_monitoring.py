import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import plotly.express as px

def cal_psi(expected_freq, observed_freq):
    """Calculate the Population Stability Index (PSI), 
    given the expected frequencies and observed frequencies.
    
    Parameters
    ----------
    expected_freq: list
        The list containing the expected frequencies.
    observed_freq: list
        The list containing the observed frequencies.

    Returns
    -------
    psi_table: pandas.core.frame.DataFrame
        Table containing the calculation of PSI value for each bin.
    """
    
    assert len(expected_freq) == len(observed_freq), 'The number of elements in both lists must be the same.'
    
    psi_table = pd.DataFrame(zip(expected_freq, observed_freq), 
                             columns=['exp_freq', 'obs_freq'])

    psi_table['exp_relative_freq'] = psi_table['exp_freq'] / psi_table['exp_freq'].sum()
    psi_table['obs_relative_freq'] = psi_table['obs_freq'] / psi_table['obs_freq'].sum()
    psi_table['diff_relative_freq'] = psi_table['exp_relative_freq'] - psi_table['obs_relative_freq'] 
    psi_table['log_e_diff_relative_freq'] = np.log(psi_table['exp_relative_freq'] / psi_table['obs_relative_freq'] )
    psi_table['psi_val'] = psi_table['diff_relative_freq'] * psi_table['log_e_diff_relative_freq']
    
    return psi_table

def get_histogram_count(expected_X, obs_X, feature_name, num_bins):
    """Calculate the expected frequencies and observed frequencies.
    
    Parameters
    ----------
    expected_X: pandas.core.frame.DataFrame
        The dataframe containing the `feature_name`'s expected values.
    obs_X: pandas.core.frame.DataFrame
        The dataframe containing the `feature_name`'s observed values.
    feature_name: str
        The feature name to obtain the expected and observed frequencies from.
    num_bins: int
        The maximum number of bins formed by the histogram. Note that that final
        bins formed may be less than `num_bins`. The num_bins will decrement by 1
        every time any one of the formed bins has less than 5 sample counts.

    Returns
    -------
    expected_freq: list
        The list containing the expected frequencies.
    observed_freq: list
        The list containing the observed frequencies.
    observed_bin_edges: numpy.ndarray
        The numpy array containing the bin edges where len(observed_bin_edges) == len(observed_freq)+1.
        The array is used to provide the bin information when displaying or visualizing the PSI value.
    """
    
    assert type(expected_X) == pd.core.frame.DataFrame, 'expected_X must be a DataFrame.'
    assert type(obs_X) == pd.core.frame.DataFrame, 'obs_X must be a DataFrame.'
    
    # Remove outliers to provide accurate results
    tmp = expected_X[feature_name]
    expected_arr = tmp[(np.abs(scipy_stats.zscore(tmp)) < 3)].copy()
    tmp = obs_X[feature_name]
    observed_arr = tmp[(np.abs(scipy_stats.zscore(tmp)) < 3)].copy()

    # Compute the histogram for both observed and expected feature values
    observed_freq, observed_bin_edges = np.histogram(observed_arr, num_bins)
    expected_freq = np.histogram(expected_arr, observed_bin_edges)[0]
    
    MIN_SAMPLES_IN_BIN = 5
    
    # Decrement number of bins by 1 and compute the histogram again
    # until all of the formed bins have 5 or more sample counts
    while observed_freq.min() < MIN_SAMPLES_IN_BIN or expected_freq.min() < MIN_SAMPLES_IN_BIN:
        num_bins -= 1
        observed_freq, observed_bin_edges = np.histogram(observed_arr, num_bins)
        expected_freq = np.histogram(expected_arr, observed_bin_edges)[0]
    
    return expected_freq, observed_freq, observed_bin_edges

def display_psi(psi_table, bin_edges=[], show_percent=False, show_sum=False):
    """Display and format the PSI values in tabular format.
    
    Parameters
    ----------
    psi_table: pandas.core.frame.DataFrame
        Table containing the calculation of PSI value for each bin.
    bin_edges: numpy.ndarray optional
        The numpy array containing the bin edges where len(observed_bin_edges) == len(observed_freq)+1.
        The array is used to provide the bin information when displaying or visualizing the PSI value.
    show_percent: bool optional
        If the show_percent is True, the expected/observed relative frequency, difference in relative 
        frequencies, and PSI value will be displayed in (%) format. These values will be displayed in 
        decimals if the show_percent is False.
    show_sum: bool optional
        If the show_sum is True, display the sum of each column at the last row of the table.
    
    Returns
    -------
    psi_table_display: pandas.core.frame.DataFrame
        Formatted table containing the calculation of PSI value for each bin.
    """
    
    assert type(psi_table) == pd.core.frame.DataFrame, 'psi_table must be a DataFrame.'
    assert type(show_percent) == bool,  'show_percent must be a boolean.'
    assert type(show_sum) == bool,  'show_sum must be a boolean.'

    no_decimal_places = 2
    
    psi_table_display = psi_table.copy()
    if show_percent:
        psi_table_display['exp_relative_freq'] *= 100
        psi_table_display['obs_relative_freq'] *= 100
        psi_table_display['diff_relative_freq'] *= 100
        psi_table_display['psi_val'] *= 100

    psi_table_display['exp_relative_freq'] = np.around(psi_table_display['exp_relative_freq'], no_decimal_places)
    psi_table_display['obs_relative_freq'] = np.around(psi_table_display['obs_relative_freq'], no_decimal_places)
    psi_table_display['diff_relative_freq'] = np.around(psi_table_display['diff_relative_freq'], no_decimal_places)
    psi_no_decimal_places = no_decimal_places if show_percent else no_decimal_places+2
    psi_table_display['psi_val'] = np.around(psi_table_display['psi_val'], psi_no_decimal_places)

    psi_table_display['log_e_diff_relative_freq'] = np.around(psi_table_display['log_e_diff_relative_freq'], 
                                                              no_decimal_places+2)
    
    if show_percent:
        psi_table_display.columns = [
            'Expected freq.', 
            'Observed freq.', 
            'Expected relative freq. (%)', 
            'Observed relative freq. (%)',
            'Diff. in relative freqs. (%)',
            'log_e_diff_relative_freq', 
            'PSI value (%)'
        ]
    else:
        psi_table_display.columns = [
            'Expected freq.', 
            'Observed freq.', 
            'Expected relative freq.', 
            'Observed relative freq.',
            'Diff. in relative freqs.',
            'log_e_diff_relative_freq', 
            'PSI value'
        ]
    
    if len(bin_edges) > 0:
        assert len(bin_edges) == len(psi_table_display)+1, \
        'The rightmost edge must be included such that len(bin_edges) == len(psi_table_display)+1'
        index_mapper = {}
        for i in range(len(bin_edges)-1):
            index_mapper[i] = f'[{round(bin_edges[i], no_decimal_places)}, '+\
                              f'{round(bin_edges[i+1], no_decimal_places)}'
            index_mapper[i] += ']' if i+1==len(bin_edges)-1 else ')'
        psi_table_display = psi_table_display.rename(index=index_mapper)

    if show_sum:
        tmp_cols = psi_table_display.columns
        tmp_index = psi_table_display.index.tolist()
        sum_ = psi_table.sum(0)
        if show_percent:
            sum_['exp_relative_freq'] *= 100
            sum_['obs_relative_freq'] *= 100
            sum_['diff_relative_freq'] *= 100
            sum_['psi_val'] *= 100
        sum_['exp_relative_freq'] = np.around(sum_['exp_relative_freq'], no_decimal_places)
        sum_['obs_relative_freq'] = np.around(sum_['obs_relative_freq'], no_decimal_places)
        sum_['diff_relative_freq'] = np.around(sum_['diff_relative_freq'], no_decimal_places)
        sum_['log_e_diff_relative_freq'] = np.around(sum_['log_e_diff_relative_freq'], no_decimal_places+2)
        sum_['psi_val'] = np.around(sum_['psi_val'], psi_no_decimal_places)
        psi_table_display = np.vstack((psi_table_display.values, sum_))
        psi_table_display = pd.DataFrame(psi_table_display.copy(), 
                                         index=tmp_index+['Total'], 
                                         columns=tmp_cols)
    
    psi_table_display = psi_table_display.astype({'Expected freq.': 'int64', 'Observed freq.': 'int64'})
    
    return psi_table_display

def determine_cal_methods(X, bin_threshold):
    """Determine if each X feature is suitable to be calculated using PSI, 
    Chi-Square Goodness of Fit Test or neither of them.
    
    Parameters
    ----------
    X: pandas.core.frame.DataFrame
        Table containing the calculation of PSI value for each bin.
    bin_threshold: int
        bin_threshold is used to decide if the nominal feature should use 
    
    Returns
    -------
    cols_calculcated_by_psi: list
        The list containing the columns that are calculcated using PSI.
    cols_calculcated_by_chi: list
        The list containing the columns that are calculcated using Chi-Square Goodness of Fit Test.
    cols_cannot_be_calculated: list
        The list containing the columns that cannot be calculcated using either PSI or 
        Chi-Square Goodness of Fit Test.
    """
    
    assert type(X) == pd.core.frame.DataFrame, 'X must be a DataFrame.'
    assert type(bin_threshold) == int, 'bin_threshold must be an integer.'
    
    cols_calculcated_by_psi = []
    cols_calculcated_by_chi = []
    cols_cannot_be_calculated = []
    
    MIN_EXPECTED_COUNTS = 5

    for feature_name, feature_values in X.iteritems():
        is_numerical = feature_values.dtype.name not in 'object'

        """
        1. Handle the case of numerical feature.
        2. Handle the case of nominal feature where the category values is not an object.
        Second condition can happened if the nomnial feature is one-hot encoded. As a 
        result, `feature_values.dtype.name` is not an 'object'.
        """
        if is_numerical:
            """
            IF the number of categories is exactly 1
                The distribution of feature is not evaluated. This can happen when the nomnial feature
                only contains one category.

            ELSE IF the nominal feature is one-hot-encoded
            
                IF smallest category's frequency count is less than 5
                    The distribution of feature is not evaluated. The frequency count must be at least 5
                    to ensure the accuracy of the Chi-Square Goodness of Fit Test.
                    
                ELSE
                    The distribution of nominal feature is checked using Chi-Square Goodness of Fit Test. 

            ELSE
                The feature is really a numerical feature.
                The distribution of numerical feature is checked using PSI. 
            """
            categories = feature_values.value_counts().index.tolist()
            if len(categories) == 1:
                cols_cannot_be_calculated.append((feature_name, 
                                                  f'Only contain 1 category which is "{categories[0]}"'))
            elif len(categories) == 2 and set(categories) == {0, 1}:
                val_count_dict = feature_values.value_counts().to_dict()
                smallest_count = min(val_count_dict.values())

                if smallest_count < MIN_EXPECTED_COUNTS:
                    smallest_cat = list(val_count_dict.keys())
                    smallest_cat = smallest_cat[list(val_count_dict.values()).index(smallest_count)]
                    cols_cannot_be_calculated.append((feature_name, 
                                                      f'The category "{smallest_cat}" only contain {smallest_count}.'))
                else:
                    cols_calculcated_by_chi.append(feature_name)
            else:
                cols_calculcated_by_psi.append(feature_name)

        # Handle the case of nominal feature
        else:
            val_counts = feature_values.value_counts()
            # Ensure that each category contains more than 5 counts
            val_counts = val_counts[val_counts >= MIN_EXPECTED_COUNTS]
            val_counts = val_counts.sort_values(ascending=True)
            cat_names = val_counts.index.tolist()
            """
            IF the number of categories >= `bin_threshold`
                The distribution of nominal feature is checked using PSI.

            ELSE
                The distribution of nominal feature is checked using Chi-Square Goodness of Fit Test. 
            """
            if len(cat_names) >= bin_threshold:
                cols_calculcated_by_psi.append(feature_name)
            else:
                cols_calculcated_by_chi.append(feature_name)

    return cols_calculcated_by_psi, cols_calculcated_by_chi, cols_cannot_be_calculated

def plot_psi_graph(psi_table, bin_edges):
    """
    Construct the PSI values visualization using Plotly.

    Parameters
    ----------
    psi_table: pandas.core.frame.DataFrame
        Table containing the calculation of PSI value for each bin.
    bin_edges: numpy.ndarray
        The numpy array containing the bin edges where len(observed_bin_edges) == len(observed_freq)+1.
        The array is used to provide the bin information when displaying or visualizing the PSI value.

    Returns
    -------
    fig: plotly.graph_objs._figure.Figure
        The Plotly Figure object.
    """

    psi_plot_data = display_psi(psi_table, bin_edges, show_percent=True, show_sum=False)
    psi_plot_data = psi_plot_data[['Expected relative freq. (%)', 'Observed relative freq. (%)']].copy()
    # psi_plot_data = pd.concat([psi_plot_data[col] for col in psi_plot_data])
    psi_plot_data_exp = pd.DataFrame(psi_plot_data['Expected relative freq. (%)'].values, 
                                    columns=['Relative Frequency (%)'], index=psi_plot_data.index)
    psi_plot_data_exp['data_type'] = 'Expected'
    psi_plot_data_obs = pd.DataFrame(psi_plot_data['Observed relative freq. (%)'].values, 
                                    columns=['Relative Frequency (%)'], index=psi_plot_data.index)
    psi_plot_data_obs['data_type'] = 'Observed'
    psi_plot_data = pd.concat([psi_plot_data_exp, psi_plot_data_obs])
    psi_plot_data = psi_plot_data.reset_index()

    colours = {
        "Expected": "#008AFA",
        "Observed": "#FF0051",
    }

    axis_color="#333333"

    fig = px.bar(psi_plot_data, 
                x='index', 
                y='Relative Frequency (%)',
                barmode='group',
                color='data_type', 
                color_discrete_map=colours,
                labels=dict(index = 'Bin', data_type='Sample Type'))

    fig.update_layout(
        width = 900,
        height = 600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis = dict(
            ticks = 'outside',
            # Set the tick color and the tick label color to lighter dark colour
            tickcolor = axis_color,
            tickfont_color = axis_color,
            showline = True,
            linewidth = 1, 
            linecolor = axis_color
        ),
        yaxis = dict(
            # Set the tick color and the tick label color to lighter dark colour
            tickcolor = axis_color,
            tickfont_color = axis_color,
            showline = True,
            linewidth = 1, 
            linecolor = axis_color
        )
    )

    fig.update_traces(
        hovertemplate = '<i>Bin:</i> %{x}<br><i>Relative Frequency</i>: %{y:.2f}<br>'
    )

    return fig

def get_category_count(expected_X, obs_X, feature_name):
    """Calculate the expected categories' count and observed categories' count.
    
    Parameters
    ----------
    expected_X: pandas.core.frame.DataFrame
        The dataframe containing the `feature_name`'s expected categories' frequency count.
    obs_X: pandas.core.frame.DataFrame
        The dataframe containing the `feature_name`'s observed categories' frequency count.
    feature_name: str
        The feature name to obtain the categories' count from.

    Returns
    -------
    exp_cat_counts: dict
        The dictionary containing the expected categories' count.
    obs_cat_counts: dict
        The dictionary containing the observed categories' count.
    """
    
    assert type(expected_X) == pd.core.frame.DataFrame, 'expected_X must be a DataFrame.'
    assert type(obs_X) == pd.core.frame.DataFrame, 'obs_X must be a DataFrame.'
    
    exp_cat_counts = expected_X[feature_name].value_counts(normalize=False).to_dict()
    obs_cat_counts = obs_X[feature_name].value_counts(normalize=False).to_dict()
    
    return exp_cat_counts, obs_cat_counts

def cal_chi2(exp_cat_counts, obs_cat_counts, feature_name, display=False):
    """Calculate the Chi-Square Goodness of Fit Test, 
    given the expected categories' count and observed categories' count.
    
    Parameters
    ----------
    exp_cat_counts: dict
        The dictionary containing the expected categories' count.
    obs_cat_counts: dict
        The dictionary containing the observed categories' count.
    feature_name: str
        The feature name to obtain the categories' count from.
    display: bool optional
        If the `display` is set to True, display the chi-squared statistics and its p-value.

    Returns
    -------
    chi2_cal_table: pandas.core.frame.DataFrame
        Table containing the calculation of Chi-Square Goodness of Fit Test for each category.
    """
    
    obs_val_counts = obs_cat_counts
    sum_exp_counts = sum(exp_cat_counts.values())
    exp_val_counts = {class_:exp_cat_counts[class_]/sum_exp_counts for class_ in exp_cat_counts.keys()}
    old_exp_val_counts = exp_cat_counts
    
    scale = sum(obs_val_counts[cat] for cat in obs_val_counts.keys())

    for cat, freq in exp_val_counts.items():
        exp_val_counts[cat] *= scale
        if cat not in obs_val_counts:
            obs_val_counts[cat] = 0

    obs_freq = [obs_val_counts[cat] for cat in sorted(obs_val_counts.keys())]
    exp_freq = [exp_val_counts[cat] for cat in sorted(exp_val_counts.keys())]
    old_exp_freq = [old_exp_val_counts[cat] for cat in sorted(old_exp_val_counts.keys())]

    chi2_cal_table = pd.DataFrame(np.array([old_exp_freq, exp_freq, obs_freq]).T, 
                                  columns=['Expected count', 'Expected count (Re-adjusted)', 'Observed count'])

    chi2_cal_table['Diff count'] = chi2_cal_table['Expected count (Re-adjusted)'] - chi2_cal_table['Observed count']
    chi2_cal_table['Squared Diff count'] = np.square(chi2_cal_table['Diff count'].values)
    chi2_cal_table['Chi2 value'] = chi2_cal_table['Squared Diff count'] / chi2_cal_table['Expected count (Re-adjusted)']

    categories = sorted(obs_val_counts.keys())
    
    if display:
        chi2_vals = chi2_cal_table['Chi2 value'].sum()
        # Get the p-value with the degree of freedom of total_number_of_categories-1
        p_val = scipy_stats.chi2.sf(chi2_vals, len(categories) - 1)
        print(f'The chi-square stats for "{feature_name}" is {round(chi2_vals, 4)} '+
              f'with the p-value of {p_val:.8g}.')

    index_mapper = {}
    for idx, cat in enumerate(categories):
        index_mapper[idx] = cat
    chi2_cal_table = chi2_cal_table.rename(index=index_mapper)
    
    return chi2_cal_table

def display_chi2(chi2_table, show_sum=False):
    """Display and format the chi2 values in tabular format.
    
    Parameters
    ----------
    chi2_table: pandas.core.frame.DataFrame
        Table containing the calculation of chi2 value for each bin.
    show_sum: bool optional
        If the show_sum is True, display the sum of each column at the last row of the table.
    
    Returns
    -------
    chi2_table_display: pandas.core.frame.DataFrame
        Formatted table containing the calculation of chi2 value for each bin.
    """
    
    assert type(chi2_table) == pd.core.frame.DataFrame, 'chi2_table must be a DataFrame.'
    assert type(show_sum) == bool,  'show_percent must be a boolean.'
    
    no_decimal_places = 2
    
    chi2_table_display = chi2_table.copy()
    
    if show_sum:
        tmp_cols = chi2_table_display.columns
        tmp_index = chi2_table_display.index.tolist()
        sum_ = chi2_table.sum(0)
        chi2_table_display = np.vstack((chi2_table_display.values, sum_))
        chi2_table_display = pd.DataFrame(chi2_table_display.copy(), 
                                         index=tmp_index+['Total'], 
                                         columns=tmp_cols)
    
    chi2_table_display['Expected count (Re-adjusted)'] = \
    np.around(chi2_table_display['Expected count (Re-adjusted)'], no_decimal_places)
    chi2_table_display['Diff count'] = \
    np.around(chi2_table_display['Diff count'], no_decimal_places)
    chi2_table_display['Squared Diff count'] = \
    np.around(chi2_table_display['Squared Diff count'], no_decimal_places)
    chi2_table_display['Chi2 value'] = \
    np.around(chi2_table_display['Chi2 value'], no_decimal_places+2)

    chi2_table_display = chi2_table_display.astype({'Expected count': 'int64', 'Observed count': 'int64'})

    return chi2_table_display
