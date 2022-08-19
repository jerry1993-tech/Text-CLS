# -*- coding: utf-8 -*-
# @Time :
# @Author :
# @File : metrics.py

import os
import sys
import numpy as np
from sklearn import metrics

local_dir = os.path.abspath('../..')
sys.path.append(local_dir)
from config import classifier_config


def cal_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    average = classifier_config['metrics_average']
    precision = metrics.precision_score(y_true, y_pred, average=average, labels=np.unique(y_pred), zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average=average, labels=np.unique(y_pred), zero_division=0)
    f1_score = metrics.f1_score(y_true, y_pred, average=average, labels=np.unique(y_pred), zero_division=0)
    each_classes = metrics.classification_report(y_true, y_pred,
                                                 output_dict=True,
                                                 labels=np.unique(y_pred),
                                                 zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}, each_classes

