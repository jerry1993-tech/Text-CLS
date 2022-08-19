# -*- coding: utf-8 -*-
# @Author : xuyingjie
# @File : data_processor.py

import sys
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from config import classifier_config


class DataManager:
    __doc__ = """the class of data processor"""

    def __init__(self, logger):
        self.logger = logger
        self.embedding_method = classifier_config['embedding_method']
        self.classifier = classifier_config['classifier']
        self.support_pretrained_model = ['Bert', 'MacBert', 'DistilBert', 'AlBert', 'Electra', 'RoBerta', 'XLNet']
        if self.embedding_method != '':
            if self.classifier in self.support_pretrained_model:
                raise Exception('如果用了预训练模型微调，则就不需引入embedding_method')

        self.remove_sp = True if classifier_config['remove_special'] else False
        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'

        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']
        self.max_sequence_len_test = classifier_config['max_sequence_len_test']

        self.classes = classifier_config['classes']
        self.class_id = {cls_name: id for id, cls_name in enumerate(self.classes)}
        self.max_label_number = len(self.classes)
        self.reverse_classes = {str(cls_id): cls_name for cls_name, cls_id in self.class_id.items()}

        self.tokenizer = self.tokenizer_for_pretrained_model(model_type=self.classifier)
        self.vocab_size = len(self.tokenizer)
        self.embedding_dim = None

    @staticmethod
    def tokenizer_for_pretrained_model(model_type):
        if model_type == 'Bert' or 'MacBert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'DistilBert':
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'AlBert':
            from transformers import AlbertTokenizer
            tokenizer = AlbertTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'Electra':
            from transformers import ElectraTokenizer
            tokenizer = ElectraTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'RoBerta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'XLNet':
            from transformers import XLNetTokenizer
            tokenizer = XLNetTokenizer.from_pretrained(classifier_config['pretrained'])
        else:
            tokenizer = None
        return tokenizer

    def prepare_pretrained_data(self, sentences, labels):
        """
        输出经预训练模型做 embedding 后的 X 矩阵和 y 矩阵
        """
        self.logger.info('loading data ...')
        tokens_list, y = [], []
        for record in tqdm(zip(sentences, labels)):
            label = tf.one_hot(record[1], depth=self.max_label_number)
            if len(record[0]) > self.max_sequence_length - 2:
                sentence = record[0][: self.max_sequence_length - 2]
                tokens = self.tokenizer.encode(sentence)
            else:
                tokens = self.tokenizer.encode(record[0])
            # 手动 padding
            if len(tokens) < self.max_sequence_length:
                tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
            tokens_list.append(tokens)
            y.append(label)

        return np.array(tokens_list), np.array(y, dtype=np.float32)

    def get_dataset(self, df):
        """
        构建 tf.dataset
        :param df: pandas.dataFrame
        :return: tf.dataset
        """
        df = df.loc[df.label.isin(self.classes)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        # 数据转成矩阵
        X, y = self.prepare_pretrained_data(df['text'], df['label'])
        # 将矩阵转成 tf.dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        return dataset

    def prepare_single_sentence(self, sentence):
        """
        将待预测的句子转成向量或矩阵
        """
        if len(sentence) > self.max_sequence_length - 2:
            sentence = sentence[:self.max_sequence_length - 2]
            tokens = self.tokenizer.encode(sentence)
        else:
            tokens = self.tokenizer.encode(sentence)
        if len(tokens) < self.max_sequence_length:
            tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]

        return np.array([tokens])


