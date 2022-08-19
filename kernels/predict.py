# -*- coding: utf-8 -*-
# @Author :
# @File : predict.py

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from config import classifier_config
from kernels.utils.cal_metrics import cal_metrics


class Predictor(object):
    def __init__(self, data_manager, logger):
        self.logger = logger
        self.data_manager = data_manager
        self.seq_length = data_manager.max_sequence_length
        self.reverse_classes = data_manager.reverse_classes
        self.checkpoints_dir = classifier_config['checkpoints_dir']

        classifier = classifier_config['classifier']
        num_classes = data_manager.max_label_number
        logger.info('loading model parameter')

        if classifier in self.data_manager.support_pretrained_model:
            from kernels.models.TFPretrainedModel import PretrainedModelClassification
            self.model = PretrainedModelClassification(num_classes, model_type=classifier)
        else:
            raise Exception('config model is not exit!')

        # 实例化 Checkpoint，设置重载对象为新训练好的模型
        checkpoint = tf.train.Checkpoint(model=self.model)
        # 从已保存文件中恢复参数
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
        logger.info('loading model successfully!')

    def predict_single(self, sentence):
        """
        单例测试
        """
        start_time = time.time()
        sentence_vec = self.data_manager.prepare_single_sentence(sentence)
        logits = self.model(inputs=sentence_vec)
        prediction = tf.argmax(logits, axis=-1)
        self.logger.info('the time consumption by prediction: %.3f(ms)' % ((time.time() - start_time)*1000))
        return self.reverse_classes[str(np.array(prediction)[0])], logits

    def predict_batch(self):
        """
        批量测试（适用于有标注的数据集）
        """
        test_file = classifier_config['test_file']
        if test_file == '':
            self.logger.info('test dataset does not exit!')
            return
        test_df = pd.read_csv(test_file)
        test_dataset = self.data_manager.get_dataset(test_df)
        batch_size = self.data_manager.batch_size
        y_true, y_pred, probabilities = np.array([]), np.array([]), np.array([])
        start_time = time.time()
        for step, batch in tqdm(test_dataset.batch(batch_size).enumerate()):
            X_test_batch, y_test_batch = batch
            logits = self.model(X_test_batch)
            predictions = tf.argmax(logits, axis=-1)
            y_test_batch = tf.argmax(y_test_batch, axis=-1)
            y_true = np.append(y_true, y_test_batch)
            y_pred = np.append(y_pred, predictions)
            max_probability = tf.reduce_max(logits, axis=-1)
            probabilities = np.append(probabilities, max_probability)
        self.logger.info('test time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        measures, each_classes = cal_metrics(y_true=y_true, y_pred=y_pred)
        # 打印总的指标
        res_str = ''
        for k, v in measures.items():
            res_str += (k + ': %.3f ' % v)
        self.logger.info('%s' % res_str)
        # 打印每一个类别的指标
        classes_val_str = ''
        for k, v in each_classes.items():
            if k in self.reverse_classes:
                classes_val_str += ('\n' + self.reverse_classes[k] + ': ' + str(each_classes[k]))
        self.logger.info(classes_val_str)

    def save_pd_model(self):
        """
        转成 pb 格式用于线上部署
        """
        tf.saved_model.save(self.model,
                            self.checkpoints_dir,
                            signatures=self.model.call.get_concrete_function(
                                tf.TensorSpec([None, self.seq_length], tf.int32, name='inputs')))

        self.logger.info('The model has been saved in pb format')
