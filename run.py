# -*- coding: utf-8 -*-
# @Time :
# @Author :
# @File : run.py

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf

local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from kernels.data_processer import DataManager
from kernels.utils.logger import get_logger
from kernels.train import Train
from config import mode, classifier_config, CUDA_VISIBLE_DEVICES
from kernels.predict import Predictor

# 设置 CPU 核数
os.environ["OMP_NUM_THREADS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed


def check(config):

    if config['checkpoints_dir'] == '':
        raise Exception('checkpoints_dir did not set...')

    if not os.path.exists(config['checkpoints_dir']):
        print('checkpoints fold not found, creating...')
        os.makedirs(config['checkpoints_dir'])

    if not os.path.exists(config['checkpoints_dir'] + '/logs'):
        print('log fold not found, creating...')
        os.mkdir(config['checkpoints_dir'] + '/logs')


if __name__ == '__main__':
    setup_seed(classifier_config['seed'])
    check(config=classifier_config)
    logger = get_logger(classifier_config['checkpoints_dir'] + '/logs', mode)
    # GPU or CPU 运行转换
    if CUDA_VISIBLE_DEVICES != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[CUDA_VISIBLE_DEVICES], True)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 训练分类器
    if mode == 'train_classifier':
        logger.info(json.dumps(classifier_config, indent=2, ensure_ascii=False))
        data_manage = DataManager(logger)
        logger.info('mode: train_classifier')
        logger.info('model: {}'.format(classifier_config['classifier']))
        train = Train(data_manage, logger)
        train.train()

    # 交互式的单例测试
    elif mode == 'predict_single':
        logger.info(json.dumps(classifier_config, indent=2, ensure_ascii=False))
        data_manager = DataManager(logger)
        logger.info('mode: predict_single')
        logger.info('model: {}'.format(classifier_config['classifier']))
        predictor = Predictor(data_manager, logger)
        predictor.predict_single('start up！')
        while True:
            logger.info('please input a sample (enter [quit] to quit.)')
            sentence = input()
            if sentence == 'quit':
                break
            result = predictor.predict_single(sentence)
            print(result)
