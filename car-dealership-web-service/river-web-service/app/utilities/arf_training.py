# Standard library & third-party libraries
from river import metrics as river_metrics

# User-defined libraries

def train_arf_rg(arf_rg, X, Y):
    """Train the adaptive random forest.

    Parameters
    ----------
    arf_rg: river.ensemble.BaseForest
        The object representing the adaptive random forest.
    X: DataFrame
        Preprocessed feature data.
    Y: np.ndarray
        Truth labels.

    Returns
    -------
    """

    data_count = 0
    # Saving metrics for evaluation purposes
    mse_metric = river_metrics.MSE()
    mae_metric = river_metrics.MAE()
    r2_metric = river_metrics.R2()
    running_metrics = {
        'mse': [],
        'mae': [],
        'r2': []
    }

    for x, y in zip(X.to_numpy(), Y):
        x = dict(zip(X.columns, x))
        # Predict current input
        y_pred = arf_rg.predict_one(x)
        # Update running metrics
        mse_metric.update(y, y_pred)
        mae_metric.update(y, y_pred)
        r2_metric.update(y, y_pred)
        if  data_count != 0 and data_count % 100 == 0:
            # Update the running metric with the prediction and ground truth value for every 100 th data
            running_metrics['mse'].append(mse_metric.get())
            running_metrics['mae'].append(mae_metric.get())
            running_metrics['r2'].append(r2_metric.get())
        arf_rg.learn_one(x, y)
        data_count += 1

    # Update running metrics for the last time
    running_metrics['mse'].append(mse_metric.get())
    running_metrics['mae'].append(mae_metric.get())
    running_metrics['r2'].append(r2_metric.get())

    return arf_rg, running_metrics

def train_arf_cf(arf_cf, X, Y):
    """Train the adaptive random forest classifier.

    Parameters
    ----------
    arf_cf: river.ensemble.BaseForest
        The object representing the adaptive random forest.
    X: DataFrame
        Preprocessed feature data.
    Y: np.ndarray
        Truth labels.

    Returns
    -------
    """

    data_count = 0
    # Saving metrics for evaluation purposes
    f1_metric = river_metrics.F1()
    precision_metric = river_metrics.Precision()
    recall_metric = river_metrics.Recall()
    roc_auc_metric = river_metrics.ROCAUC()
    running_metrics = {
        'f1_score': [],
        'precision': [],
        'recall': [],
        'roc_auc': []
    }

    for x, y in zip(X.to_numpy(), Y):
        x = dict(zip(X.columns, x))
        # Predict current input
        y_pred = arf_cf.predict_one(x)
        y_pred_proba = arf_cf.predict_proba_one(x)
        # Update running metrics
        f1_metric.update(y, y_pred)
        precision_metric.update(y, y_pred)
        recall_metric.update(y, y_pred)
        roc_auc_metric.update(y, y_pred_proba)
        if  data_count != 0 and data_count % 100 == 0:
            # Update the running metric with the prediction and ground truth value for every 100 th data
            running_metrics['f1_score'].append(f1_metric.get())
            running_metrics['precision'].append(precision_metric.get())
            running_metrics['recall'].append(recall_metric.get())
            running_metrics['roc_auc'].append(roc_auc_metric.get())
        arf_cf.learn_one(x, y)
        data_count += 1

    # Update running metrics for the last time
    running_metrics['f1_score'].append(f1_metric.get())
    running_metrics['precision'].append(precision_metric.get())
    running_metrics['recall'].append(recall_metric.get())
    running_metrics['roc_auc'].append(roc_auc_metric.get())

    return arf_cf, running_metrics