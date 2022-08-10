# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from itertools import product, combinations

# scikit-learn metrics
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

DECIMAL_POINTS = 3

def arf_predict_proba(arf_cf, X):
    """Perform predictions using an adaptive random forest model.
    
    Parameters
    ----------
    arf_cf: river.ensemble.BaseForest
        The object representing the adaptive random forest.
    X: DataFrame
        Feature values

    Returns
    -------
    y_pred_probas: list
        Predicted probabilities for positive and negative classes.
    """
    y_pred_probas = []
    
    desc = f'Performing predictions'
    with tqdm(total=X.shape[0], position=0, leave=True, desc=desc) as pbar:
        progress_unit = 1
        for x in X.to_numpy():
            x = dict(zip(X.columns, x))
            # Predict current input
            y_pred_proba = arf_cf.predict_proba_one(x)
            y_pred_probas.append([y_pred_proba[0], y_pred_proba[1]])
            # Update the progress bar.
            pbar.update(progress_unit)
        
    return np.array(y_pred_probas, dtype=np.float64)

def arf_predict(arf_cf, X):
    """Perform predictions using an adaptive random forest model.
    
    Parameters
    ----------
    arf_cf: river.ensemble.BaseForest
        The object representing the adaptive random forest.
    X: DataFrame
        Feature values

    Returns
    -------
    y_preds: list
        Predicted classes.
    y_pred_probas: list
        Predicted probabilities for positive classes.
    """
    y_preds = []
    y_pred_probas = []
    
    desc = f'Performing predictions'
    with tqdm(total=X.shape[0], position=0, leave=True, desc=desc) as pbar:
        progress_unit = 1
        for x in X.to_numpy():
            x = dict(zip(X.columns, x))
            # Predict current input
            y_pred = arf_cf.predict_one(x)
            y_pred_proba = arf_cf.predict_proba_one(x)[1.0]
            y_preds.append(y_pred)
            y_pred_probas.append(y_pred_proba)
            # Update the progress bar.
            pbar.update(progress_unit)
        
    return [y_preds, y_pred_probas]

def agg_cf_res(pred_records, y, digits):
    """Calculate the performance metrics given the predictions and the truth labels.
    
    Then, format and aggregate the models' performance metrics into a single table.
    Finally, The summary table is grouped by metric type for easier viewing.
    
    Parameters
    ----------
    pred_records: dict
        The dictionary containing the models' predictions. 
        The keys are the model names and the values are the models' predictions, respectively.
    y: np.ndarray
        Class labels.
    digits: int, optional
        Number of decimal points to be kept in float values.

    Returns
    -------
    res_df: DataFrame
        Table that summarised the models' performance.
    """
    cf_results = []

    # Call classification_report from scikit-learn API to get performance metrics
    # The API will return a dictionary containing the metrics
    model_types = pred_records.keys()
    for model_type in model_types:
        cf_results.append(classification_report(y, pred_records[model_type], output_dict=True))
    
    # Remove less useful information
    ACC = 'accuracy'
    for cf_res in cf_results:
        if ACC in cf_res:
            del cf_res[ACC]
    
    indexes = list(cf_results[0].keys())
    metric_types = list(cf_results[0][indexes[0]].keys())
    column_names = list(product(metric_types, model_types))
    stride = len(model_types)
    num_rows = len(indexes)
    num_cols = len(metric_types) * stride
    res_arr = np.empty((num_rows, num_cols))

    # Transfer the metrics from the returned dictionaries, 
    # then aggregate the information into a summmary table
    for row_idx, category in enumerate(indexes):
        for col_idx, metric_name in enumerate(metric_types):
            metrics = []
            for cf_res in cf_results:
                # Format the metric values
                metrics.append(round(cf_res[category][metric_name], digits))
            for idx, model_type in enumerate(model_types):
                res_arr[row_idx, col_idx*stride+idx] = metrics[idx]

    res_df = pd.DataFrame(res_arr, index=indexes, columns=column_names)
    res_df.columns = pd.MultiIndex.from_tuples(res_df.columns, names=['Metrics', 'Model type'])
    
    # Display the summary table
    print(f'\nClassification results summary table by model type:')
    display(res_df)
    
    # Break down the summary table by metric type for easier viewing
    # Get the combination of model pairs
    model_pairs = list(combinations(model_types, 2))
    
    for metric_type in ['precision', 'recall', 'f1-score']:
        print(f'\nComparing {metric_type}:')
        res_df_tmp = res_df[metric_type].copy(deep=True)
        for model_type1, model_type2, in model_pairs:
            # Create a column that display the performance difference between a pair of models
            col_name = f'{model_type1} - {model_type2}'
            res_df_tmp[col_name] = res_df_tmp[model_type1] - res_df_tmp[model_type2]
        display(res_df_tmp)
        
    return res_df

def plot_roc_curves(pred_records, plot_title, xlabel, ylabel, digits=DECIMAL_POINTS, figsize=(15, 8)):
    """Plot the ROC curves given the predictions and the truth labels.
    
    Parameters
    ----------
    pred_records: dict
        The dictionary containing the models' predictions, predicted probabilities and more.
        The dictionary structure is shown below.
        {
            'pred/fpr/tpr/roc_auc/label/pred_proba': {
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
    # Visualization settings - https://stackoverflow.com/a/39566040
    plt.figure(figsize=figsize)
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
    
    model_types = pred_records['pred_proba'].keys()
    
    for model_type in model_types:
        # Retrieve information from the record
        y_pred_proba = pred_records['pred_proba'][model_type]
        fpr = pred_records['fpr'][model_type]
        tpr = pred_records['tpr'][model_type]
        roc_auc = pred_records['roc_auc'][model_type]
        label = pred_records['label'][model_type]
        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'{label} (AUC={round(roc_auc, digits)})')
        
    plt.xlabel(xlabel, fontsize=MEDIUM_SIZE)
    # Set the y axis label of the current axis.
    plt.ylabel(ylabel, fontsize=MEDIUM_SIZE)
    # Set a title of the current axes.
    plt.title(plot_title, fontsize=BIGGER_SIZE)
    # Show a legend on the plot
    plt.legend(loc='lower right')
    # Set xlimit and ylimit
    plt.xlim([0, 1])
    plt.xlim([0, 1])
    # Display a figure.
    plt.show()
    
def measure_performance(pred_records, X, y, data_pp, plot_title):
    """Measure and compare classification performance between traditional random forest and adaptive random forest.
    
    Parameters
    ----------
    pred_records: dict
        The dictionary containing the models' predictions, predicted probabilities and more.
        The dictionary structure is shown below.
        {
            'pred/fpr/tpr/roc_auc/label/pred_proba': {
                'ARF/TRF': 
            }
        }
    X: DataFrame
        Feature values.
    y: np.ndarray
        Class labels.
    trf: sklearn.ensemble.RandomForestClassifier
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
    X_pp = data_pp.preprocess(X)
    model_types = pred_records['model'].keys()

    # Get predicted class and predicted probability for positive class for each sample
    for model_type in model_types:
        model = pred_records['model'][model_type]
        # TRF
        if isinstance(model, RandomForestClassifier):
            pred_records['pred'][model_type] = model.predict(X_pp)
            pred_records['pred_proba'][model_type] = model.predict_proba(X_pp)[:, 1]
        # ARF
        else:
            y_preds_arf = arf_predict(model, X_pp)
            pred_records['pred'][model_type] = y_preds_arf[0]
            pred_records['pred_proba'][model_type] = y_preds_arf[1]
    
    # Calculate ROCs and AUC
    for model_type in model_types:
        y_pred_proba = pred_records['pred_proba'][model_type]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        # Save the result for visualization
        pred_records['fpr'][model_type] = fpr
        pred_records['tpr'][model_type] = tpr
        pred_records['roc_auc'][model_type] = roc_auc
        pred_records['label'][model_type] = f'{model_type}\'s ROC Curve'
        
    params = {
        'pred_records': pred_records['pred'],
        'y': y,
        'digits': DECIMAL_POINTS
    }

    # Visualise the models' performance in table form
    res_df = agg_cf_res(**params)
        
    plot_config = {
        'pred_records': pred_records,
        'plot_title': plot_title,
        'xlabel': 'False positive rate',
        'ylabel': 'True positive rate',
        'digits': DECIMAL_POINTS
    }

    # Visualise the models' performance in graph
    plot_roc_curves(**plot_config)

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
            if isinstance(model, RandomForestClassifier):
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

def get_cf_metrics(y_true, y_pred, y_proba):
    """Calculate the classification metrics.

    Parameters
    ----------
    y_true: np.ndarray
        Truth labels.
    y_pred: np.ndarray
        Predictions.
    y_proba: np.ndarray
        Predicted probabilities.

    Returns
    -------
    res: dict
        Dictionary containing different classification metrics.
    """
    res = {}
    res['f1_score'] = f1_score(y_true, y_pred)
    res['precision'] = precision_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred)
    res['roc_auc'] = roc_auc_score(y_true, y_proba)
    return res

def get_running_metrics(running_metrics, pred_records, model_type, y_true, data_interval):
    """Calculate and update running metrics for every fixed number of samples.
    
    Parameters
    ----------
    running_metrics: dict
        Dictionary containing running metrics.
    pred_records: dict
        The dictionary containing the models' predictions and predicted probabilities .
        The dictionary structure is shown below.
        {
            'pred/pred_proba': {
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
    
    y_pred_proba = pred_records['pred_proba'][model_type]
    y_pred = pred_records['pred'][model_type]

    # Get the stop indexes e.g [0, 100, 200, ..., 24900, 25000]
    stop_indexes = np.arange(data_interval, len(y_pred) + 1, data_interval)

    desc = f'Calculating running metrics for {model_type}'

    with tqdm(total=len(stop_indexes), position=0, leave=True, desc=desc) as pbar:
        progress_unit = 1
        
        for stop_idx in stop_indexes:
            # Calculate the performance metrics
            cf_res = get_cf_metrics(y_true[:stop_idx], y_pred[:stop_idx], y_pred_proba[:stop_idx])
            # Update the performance metrics
            for metric_name in running_metrics:
                running_metrics[metric_name].append(cf_res[metric_name])

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
        plt.legend(loc='lower left')
        # Set the x axis label
        plt.xlabel(f'Training\'s batch (by {data_interval}th)', fontsize=MEDIUM_SIZE)
        # Set the y axis label of the current axis.
        plt.ylabel('Performance', fontsize=MEDIUM_SIZE)
        # Set a title of the current axes.
        plt.title(f'Running {metric_type} of models versus training time', fontsize=BIGGER_SIZE)

    plt.show()