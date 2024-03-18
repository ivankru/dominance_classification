import numpy as np
from typing import List, Any

def calculate_metrics(y_true: Any, y_pred: Any, conf_thr: float = .5, metrics: List[float] = [],
                       metrics_general: List[float] = []) -> dict:
    '''
    Function for calculating metrics
    y_true - true labels
    y_pred - predicted class probabilities
    conf_thr - threshold at which an instance belongs to a class
    metrics - metrics that can be calculated separately for the left, 
              separately for the right type of blood circulation
    metrics_general - metrics that are counted for the whole datasets

    return: dict with the metrics
    '''

    dict_metrics = dict()
    
    y_true_r = y_true
    y_pred_r = y_pred[:, 1] > conf_thr
    y_true_l = np.logical_not(y_true)
    y_pred_l = y_pred[:, 0] > conf_thr
    
    for metric in metrics:
        print(str(metric).split()[1], 'left', metric(y_true_l, y_pred_l))
        print(str(metric).split()[1], 'right', metric(y_true_r, y_pred_r))

        dict_metrics[str(metric).split()[1]+' '+'left'] = [metric(y_true_l, y_pred_l)]
        dict_metrics[str(metric).split()[1]+' '+'right'] = [metric(y_true_r, y_pred_r)]

    for metric in metrics_general:
        print(str(metric).split()[1], metric(y_true_r, y_pred_r))

        dict_metrics[str(metric).split()[1]] = [metric(y_true_r, y_pred_r)]
    
    return dict_metrics
