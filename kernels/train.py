# -*- coding: utf-8 -*-
# @Author :
# @File : train.py

import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from config import classifier_config
from kernels.utils.cal_metrics import cal_metrics
tf.keras.backend.set_floatx('float32')
tf.get_logger().setLevel('ERROR')


class Train:
    def __init__(self, data_manager, logger):
        self.logger = logger
        self.data_manager = data_manager
        self.reverse_classes = data_manager.reverse_classes

        self.epoch = classifier_config['epoch']
        self.print_per_batch = classifier_config['print_per_batch']
        self.is_early_stop = classifier_config['is_early_stop']
        self.patient = classifier_config['patient']
        self.batch_size = data_manager.batch_size

        self.loss_function = CategoricalCrossentropy()

        learning_rate = classifier_config['learning_rate']
        weight_decay = classifier_config['weight_decay']
        classifier = classifier_config['classifier']
        num_classes = data_manager.max_label_number

        if classifier_config['optimizer'] == 'Adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'Adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'AdamW':
            from tensorflow_addons.optimizers import AdamW
            self.optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception('optimizer does not exist')

        # 载入模型
        if classifier in self.data_manager.support_pretrained_model:
            from kernels.models.TFPretrainedModel import PretrainedModelClassification
            self.model = PretrainedModelClassification(num_classes, model_type=classifier)
        else:
            raise Exception('config model is not exist')

        checkpoints_dir = classifier_config['checkpoints_dir']
        checkpoint_name = classifier_config['checkpoint_name']
        max_to_keep = classifier_config['max_to_keep']

        my_checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=my_checkpoint, directory=checkpoints_dir,
            checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
        my_checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('restored from {}'.format(self.checkpoint_manager.latest_checkpoint))
        else:
            print('initializing from scratch.')

    def train(self):
        train_file = classifier_config['train_file']
        val_file = classifier_config['val_file']
        set_val_rate = classifier_config['set_val_rate']
        train_df = pd.read_csv(train_file, sep="\t").sample(frac=1)
        # 部分样本训练
        train_df = train_df.sample(frac=1)[:10000]

        if val_file == '' and set_val_rate:
            self.logger.info('need to generate validation dataset...')
            ratio = 1 - set_val_rate
            # 划分训练集和验证集
            train_df, val_df = train_df[:int(len(train_df) * ratio)], train_df[int(len(train_df) * ratio):]
            val_df = val_df.sample(frac=1)
        else:
            val_df = pd.read_csv(val_file).sample(frac=1)

        train_dataset = self.data_manager.get_dataset(train_df)
        val_dataset = self.data_manager.get_dataset(val_df)

        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0
        very_start_time = time.time()

        self.logger.info(('=' * 20) + 'training starting' + ('=' * 20))
        for i in range(self.epoch):
            start_time = time.time()
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            for step, batch in tqdm(train_dataset.shuffle(len(train_df)).batch(self.batch_size).enumerate()):
                X_train_batch, y_train_batch = batch
                with tf.GradientTape() as tape:
                    logits = self.model(X_train_batch, training=1)
                    loss_vec = self.loss_function(y_true=y_train_batch, y_pred=logits)
                    loss = tf.reduce_sum(loss_vec)
                # 定义好梯度的参数
                variables = self.model.trainable_variables
                # 将预训练模型里面的 pooler 层的参数去掉
                variables = [var for var in variables if 'pooler' not in var.name]
                gradients = tape.gradient(loss, variables)

                # 反向传播，自动微分计算
                self.optimizer.apply_gradients(zip(gradients, variables))
                if step % self.print_per_batch == 0 and step != 0:
                    predictions = tf.argmax(logits, axis=-1).numpy()
                    y_train_batch = tf.argmax(y_train_batch, axis=-1).numpy()
                    measures, _ = cal_metrics(y_true=y_train_batch, y_pred=predictions)
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f' % v)
                    self.logger.info('training batch: %5d, loss: %.5f, %s' % (step, loss, res_str))

            # 每跑完一个epoch就验证一下
            measures = self.validate(val_dataset)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)

            if measures['f1_score'] > best_f1_val:
                unprocessed = 0
                best_f1_val = measures['f1_score']
                best_at_epoch = i + 1
                self.checkpoint_manager.save()
                self.logger.info('Saved the new best model with f1_score: %.3f' % best_f1_val)
            else:
                unprocessed += 1

            if self.is_early_stop:
                if unprocessed >= self.patient:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                    return

        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    def validate(self, val_dataset):
        self.logger.info('start evaluate engines...')
        y_true, y_pred = np.array([]), np.array([])
        loss_values = []
        for val_batch in tqdm(val_dataset.batch(self.batch_size)):
            X_val_batch, y_val_batch = val_batch
            logits = self.model(X_val_batch)
            val_loss_vec = self.loss_function(y_true=y_val_batch, y_pred=logits)
            val_loss = tf.reduce_mean(val_loss_vec)
            predictions = tf.argmax(logits, axis=-1)
            y_val_batch = tf.argmax(y_val_batch, axis=-1)
            y_true = np.append(y_true, y_val_batch)
            y_pred = np.append(y_pred, predictions)
            loss_values.append(val_loss)

        measures, each_classes = cal_metrics(y_true=y_true, y_pred=y_pred)

        # 打印每一个类的指标
        classes_val_str = ''
        for k, v in each_classes.items():
            if k in self.reverse_classes:
                classes_val_str += ('\n' + str(self.reverse_classes[k]) + ': ' + str(each_classes[k]))
        self.logger.info(classes_val_str)
        # 打印损失函数
        val_res_str = 'loss: %.3f ' % np.mean(loss_values)
        for k, v in measures.items():
            val_res_str += (k + ': %.3f ' % measures[k])
        self.logger.info(val_res_str)

        return measures


