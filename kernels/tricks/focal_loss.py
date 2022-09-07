# -*- coding: utf-8 -*-
# @Author : xuyingjie
# @File : focal_loss.py

import os
import sys
local_dir = os.path.abspath('../..')
sys.path.append(local_dir)
import tensorflow as tf
from config import classifier_config


class FocalLoss(tf.keras.losses.Loss):
    __doc__ = """适用于多分类的focal loss"""

    def __init__(self, gamma=2.0, epsilon=1e-9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # 针对多分类的不平衡
        weight = classifier_config['weight']
        self.alpha = tf.reshape(weight, [-1]) if weight else None
        self.epsilon = epsilon
        self.num_labels = len(classifier_config['classes'])

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        labels = tf.expand_dims(tf.argmax(y_true, axis=-1), axis=-1)
        print("labels=", labels)
        pt = tf.gather(y_pred, labels, batch_dims=1)
        print("pt=", pt)
        pt = tf.clip_by_value(pt, self.epsilon, 1. - self.epsilon)
        loss_vec = -tf.multiply(tf.pow(tf.subtract(1., pt), self.gamma), tf.math.log(pt))
        if self.alpha is not None:
            alpha = tf.gather(self.alpha, labels, batch_dims=0)
            loss_vec = tf.multiply(alpha, loss_vec)
        return loss_vec

