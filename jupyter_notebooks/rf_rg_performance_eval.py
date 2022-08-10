# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline

# scikit-learn metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DECIMAL_POINTS = 4

def arf_predict(arf_rg, X):
    """Perform predictions using an adaptive random forest model.
    
    Parameters
    ----------
    arf_rg: river.ensemble.BaseForest
        The object representing the adaptive random forest.
    X: DataFrame
        Feature values

    Returns
    -------
    y_preds: list
        Predicted output.
    """
    y_preds = []
    
    desc = f'Performing predictions'
    with tqdm(total=X.shape[0], position=0, leave=True, desc=desc) as pbar:
        progress_unit = 1
        for x in X.to_numpy():
            x = dict(zip(X.columns, x))
            # Predict current input
            y_pred = arf_rg.predict_one(x)
            y_preds.append(y_pred)
            # Update the progress bar.
            pbar.update(progress_unit)
        
    return y_preds

def agg_rg_res(pred_records, digits):
    """Calculate the performance metrics given the predictions and the truth labels.
    
    Then, format and aggregate the models' performance metrics into a single table.
    
    Parameters
    ----------
    pred_records: dict
        The dictionary containing the models' predictions.
        The dictionary structure is shown below.
        {
            'residual/pred/mae/mse/r2/label': {
                'ARF/TRF': 
            }
        }
    digits: int, optional
        Number of decimal points to be kept in float values.

    Returns
    -------
    res_df: DataFrame
        Table that summarised the models' performance.
    """

    # Call classification_report from scikit-learn API to get performance metrics
    # The API will return a dictionary containing the metrics
    model_types = pred_records['model'].keys()
    metric_types = ['mse', 'mae', 'r2']

    res_arr = {model_type: [] for model_type in model_types}

    for model_type in model_types:
        for metric_type in metric_types:
            res_arr[model_type].append(round(pred_records[metric_type][model_type], digits))

    res_df = pd.DataFrame(res_arr, index=metric_types)

    # Break down the summary table by metric type for easier viewing
    # Get the combination of model pairs
    model_pairs = list(combinations(model_types, 2))
    
    for metric_type in metric_types:
        for model_type1, model_type2, in model_pairs:
            # Create a column that display the performance difference between a pair of models
            col_name = f'{model_type1} - {model_type2}'
            res_df[col_name] = np.round(res_df[model_type1] - res_df[model_type2], digits)
 
    display(res_df)

    return res_df

def make_residuals_plot(pred_records, binwidth, plot_title, digits, figsize=(12, 8)):
    """Plot the residuals given the predictions and the truth labels.
    
    Parameters
    ----------
    pred_records: dict
        The dictionary containing the models' predictions and more.
        The dictionary structure is shown below.
        {
            'residual/pred/mae/mse/r2/label': {
                'ARF/TRF': 
            }
        }
    plot_title: str
        The title of the Matplotlib plot.
    xlabel: str
        The label for x-axis.
    ylabel: str
        The label for y-axis.
    digits: int, optional
        Number of decimal points to be kept in float values.
    figsize: tuple of int, optional
        Size of the plot indicated by (width, height).

    Returns
    -------

    """
    # Configuration
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # colors = ['#B7D299', '#3F93B7']
    
    # start with a square Figure
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(1, 2, width_ratios=(8, 2), wspace=0.05)

    # Define main plot
    ax = fig.add_subplot(gs[0, 0], xlabel='Predicted values', ylabel='Residuals')
    # Define side plot
    ax_hist = fig.add_subplot(gs[0, 1], xlabel='Distribution', yticks=[], 
                              yticklabels=[])
    
    model_types = list(pred_records['model'].keys())
    
    # Scatter plot at the main plot
    for model_type in model_types:
        ax.scatter(pred_records['pred'][model_type], pred_records['residual'][model_type], 
                   label=pred_records['label'][model_type], alpha=0.7)
    
    # Set plot title
    ax.set_title(plot_title, fontsize=BIGGER_SIZE)
    
    # Activate the legend
    ax.legend(loc='upper left')
    
    # Draw a horizontal line at y=0
    ax.axhline(y=0, color='black', alpha=0.8, linestyle='-')
    
    # Add grid
    ax.grid()

    # Determine the limits
    # Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    x = pred_records['pred'][model_types[0]]
    y = pred_records['residual'][model_types[0]]
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)

    for model_type in model_types:
        ax_hist.hist(pred_records['residual'][model_type], bins=bins, density=True, 
                    orientation='horizontal', alpha=0.7)
        
    plt.show()

def measure_performance(pred_records, X, y, data_pp=None, plot_title='Performance plot', binwidth=1):
    """Measure and compare regression performance between traditional random forest and adaptive random forest.
    
    Parameters
    ----------
    pred_records: dict
        The dictionary containing the models' predictions and more.
        The dictionary structure is shown below.
        {
            'pred/residual/mae/mse/r2/label': {
                'ARF/TRF': 
            }
        }
    X: DataFrame
        Feature values.
    y: np.ndarray
        Class labels.
    trf: sklearn.ensemble.RandomForestRegressor
        Object representation of the traditional random forest.
    arf: river.ensemble.BaseForest
        Object representation of the adaptive random forest.
    data_pp: DataPreprocessor
        Data preprocessor.
    plot_title: str
        The title of the Matplotlib plot.

    Returns
    -------
    """
    # Preprocess data
    if data_pp is not None:
        X = data_pp.preprocess(X)
    model_types = pred_records['model'].keys()

    # Get predictions for each sample
    for model_type in model_types:
        model = pred_records['model'][model_type]
        # TRF
        if isinstance(model, RandomForestRegressor) or \
           isinstance(model, SklearnPipeline):
            pred_records['pred'][model_type] = model.predict(X)
        # ARF
        else:
            pred_records['pred'][model_type] = arf_predict(model, X)
        pred_records['residual'][model_type] = y - pred_records['pred'][model_type]

    # Calculate ROCs and AUC
    for model_type in model_types:
        # Save the result for visualization
        mse = mean_squared_error(y, pred_records['pred'][model_type])
        mae = mean_absolute_error(y, pred_records['pred'][model_type])
        r2 = r2_score(y, pred_records['pred'][model_type])
        pred_records['mse'][model_type] = mse
        pred_records['mae'][model_type] = mae
        pred_records['r2'][model_type] = r2
        pred_records['label'][model_type] = f'{model_type}\'s R2: {round(r2, DECIMAL_POINTS)}'
        
    params = {
        'pred_records': pred_records,
        'digits': DECIMAL_POINTS
    }

    # Visualise the models' performance in table form
    res_df = agg_rg_res(**params)

    plot_config = {
        'pred_records': pred_records,
        'binwidth': binwidth,
        'plot_title': plot_title,
        'digits': DECIMAL_POINTS
    }

    # Visualise the models' performance in graph
    make_residuals_plot(**plot_config)

def analyze_trees(models):
    """Analyze the tree structure given the ensemble objects. 
    The supported ensemble objects are sklearn.ensemble.RandomForestClassifier and 
    river.ensemble.AdaptiveRandomForestClassifier.
    
    Parameters
    ----------
    models: dict
        The dictionary containing the models.
        The directory structure should look the one below.
        {
            'model_name': <model_object>, ...
        }

    Returns
    -------
    """

    model_types = list(models.keys())
    num_base_learner = len(models[model_types[0]])
    indexes = list(product(model_types, ['total_nodes', 'max_height']))
    index = pd.MultiIndex.from_tuples(indexes, names=['Model', 'Properties'])
    tree_summary = pd.DataFrame(index=index, columns=[i+1 for i in range(num_base_learner)])

    for idx in range(num_base_learner):
        for model_type, model in models.items():
            if isinstance(model, RandomForestRegressor):
                tree_summary.loc[(model_type, 'total_nodes'), idx+1] = \
                model.estimators_[idx].tree_.node_count
                
                tree_summary.loc[(model_type, 'max_height'), idx+1] = \
                model.estimators_[idx].tree_.max_depth
                
            else:
                tree_summary.loc[(model_type, 'total_nodes'), idx+1] = \
                model[idx].model.n_nodes

                # Substract the height by 1, the deepest node should start with height 0
                # But the River API source code start counting height from 1 as shown
                # in this link https://github.com/online-ml/river/blob/main/river/tree/base.py#L182
                tree_summary.loc[(model_type, 'max_height'), idx+1] = \
                model[idx].model.height - 1

    display(tree_summary)

    print("\nTotal nodes:\n")

    display(tree_summary.xs(('total_nodes'), level=('Properties')))

    print("\nMaximum height:\n")

    display(tree_summary.xs(('max_height'), level=('Properties')))

def get_rg_metrics(y_true, y_pred):
    """Calculate the regression metrics.

    Parameters
    ----------
    y_true: np.ndarray
        Truth labels.
    y_pred: np.ndarray
        Predictions.

    Returns
    -------
    res: dict
        Dictionary containing different regression metrics.
    """
    res = {}

    res['mse'] = mean_squared_error(y_true, y_pred)
    res['mae'] = mean_absolute_error(y_true, y_pred)
    res['r2'] = r2_score(y_true, y_pred)

    return res

def get_running_metrics(running_metrics, pred_records, model_type, y_true, data_interval):
    """Calculate and update running metrics for every fixed number of samples.
    
    Parameters
    ----------
    running_metrics: dict
        Dictionary containing running metrics.
    pred_records: dict
        The dictionary containing the models' predictions.
        The dictionary structure is shown below.
        {
            'pred': {
                'ARF/TRF': 
            }
        }
    model_type: {'ARF', 'TRF'}
        The model type.
    y_true: np.ndarray
        Truth labels.
    data_interval: int
        The data interval/stride for each metrics' calculation oand update.
    
    Returns
    -------
    """
    
    y_pred = pred_records['pred'][model_type]

    # Get the stop indexes e.g [0, 100, 200, ..., 24900, 25000]

    # E.g. 4128 is "round up" to 4200 if data_interval is 100,
    # make sure the stop index is divisible by data_interval
    final_stop_index = len(y_pred)
    left_over = final_stop_index % data_interval
    if left_over > 0:
        final_stop_index = final_stop_index - left_over + data_interval

    stop_indexes = np.arange(data_interval, final_stop_index + 1, data_interval)

    desc = f'Calculating running metrics for {model_type}'

    with tqdm(total=len(stop_indexes), position=0, leave=True, desc=desc) as pbar:
        progress_unit = 1
        
        for stop_idx in stop_indexes:
            # Calculate the performance metrics
            rg_res = get_rg_metrics(y_true[:stop_idx], y_pred[:stop_idx])
            # Update the performance metrics
            for metric_name in running_metrics:
                running_metrics[metric_name].append(rg_res[metric_name])

            # Update the progress bar.
            pbar.update(progress_unit)

def plot_performance_graph(metrics_list, metric_type, y_train_size, y_test_size, data_interval, 
                           text_x_positions, text_y_position, ylimit, figure=(15, 8)):
    """Visualize multiple models' running metrics on a graph. 
       
    Parameters
    ----------
    metrics_list: dict
        Dictionary where each item has the running metrics for a model.
        The dictionary structure is shown below.
        {
            'model_name1': {
                'f1_score/roc_auc/precision/recall': 
            }, ...
        }
    metric_type: str
        The type of the running metrics to be displayed.
    y_train_size: int
        The number of each normal/drifted train target.
    y_test_size: int
        The number of each normal/drifted test target.
    data_interval: int
        The data interval/stride for each metrics' calculation oand update.
    text_x_positions: list of int
        The x position of the respective texts.
    text_y_position: int
        The y position of the texts.
    ylimit: list of int
        The list containing the start and end value of the y-axis.
    figure: tuple of int
        The tuple containing the width and height of the plot.
    
    Returns
    -------
    """
    # Visualization settings - https://stackoverflow.com/a/39566040
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=figure)

    # Calculate the partition size
    train_par_size = int(y_train_size / data_interval)
    if y_train_size % data_interval > 0:
        train_par_size += 1
    test_par_size = int(y_test_size / data_interval)
    if y_test_size % data_interval > 0:
        test_par_size += 1

    partitions = [
        train_par_size, train_par_size+test_par_size, 
        train_par_size*2+test_par_size, train_par_size*2+test_par_size*2
    ]
    labels = ['Offline train data', 'Offline test data', 'Online train data', 'Online test data']

    for idx in range(len(text_x_positions)):    
        partition = partitions[idx]
        text_x_position = text_x_positions[idx]
        label = labels[idx]
        # plot comparison lines
        plt.axvline(x = partition, color='dimgray', linestyle='dashed')
        plt.text(text_x_position*partition, text_y_position, label, rotation = 0)

    metrics_list_key = list(metrics_list.keys())
    x_data = [i for i in range(len(metrics_list[metrics_list_key[0]][metric_type]))]
    # plotting the metric scores
    for idx, (model_name, metrics_record) in enumerate(metrics_list.items()):
        plt.plot(x_data, metrics_record[metric_type], label = f"{model_name}")
        # Set xlimit
        plt.xlim([1, partitions[-1]])
        # Set ylimit
        plt.ylim(ylimit)
        # show a legend on the plot
        plt.legend(loc='lower right')
        # Set the x axis label
        plt.xlabel(f'Training\'s batch (by {data_interval}th)', fontsize=MEDIUM_SIZE)
        # Set the y axis label of the current axis.
        plt.ylabel('Performance', fontsize=MEDIUM_SIZE)
        # Set a title of the current axes.
        plt.title(f'Running {metric_type} of models versus training time', fontsize=BIGGER_SIZE)

    plt.show()