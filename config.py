# -*- coding: utf-8 -*-
# @Author : xuyingjie
# @File : config.py


# [train_classifier, predict_single, predict_test, save_pb_model]
mode = 'train_classifier'

CUDA_VISIBLE_DEVICES = -1
# int, -1:CPU, [0,..]:GPU
# coincides with tf.CUDA_VISIBLE_DEVICES

classifier_config = {
    # 模型选择
    # 预训练模型：Bert/MacBert/RoBerta/DistilBert/AlBert/Electra/XLNet
    'classifier': 'MacBert',
    # 若选择Bert系列微调做分类，请在pretrained指定预训练模型的版本
    'pretrained': '/Users/xuyingjie/PycharmProjects/Tensorflow_Pytorch_NLP/Text_CLS/pre_models/chinese_macbert_base',
    # 训练数据集
    'train_file': '/Users/xuyingjie/PycharmProjects/Tensorflow_Pytorch_NLP/Text_CLS/data/训练集/train.csv',
    # 验证数据集
    'val_file': '',
    'set_val_rate': 0.15,
    # 测试数据集
    'test_file': '/Users/xuyingjie/PycharmProjects/Tensorflow_Pytorch_NLP/Text_CLS/data/测试集/test1.csv',
    'test_file2': '/Users/xuyingjie/PycharmProjects/Tensorflow_Pytorch_NLP/Text_CLS/data/提交示例.csv',
    # 引入外部的词嵌入,可选word2vec、Bert
    # word2vec:使用word2vec词向量做特征增强
    # 不填写则随机初始化的Embedding
    'embedding_method': '',
    # token的粒度,token选择字粒度的时候，词嵌入(embedding_method)无效
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'word',
    # 是否去掉特殊字符
    'remove_special': True,
    # 类别列表
    'classes': [0, 1],
    # 模型保存的文件夹
    'checkpoints_dir': '/Users/xuyingjie/PycharmProjects/Tensorflow_Pytorch_NLP/Text_CLS/checkpoints_trained/MacBert_base_model',
    # 模型保存的名字
    'checkpoint_name': 'MacBert_base_model',
    # 设置随机种子便于结果复现
    'seed': 1024,
    # 学习率
    # 微调预训练模型时建议更小，设置5e-5
    'learning_rate': 4e-5,
    # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    'weight_decay': 1e-2,
    # 优化器选择
    # 可选：Adagrad/Adadelta/RMSprop/SGD/Adam/AdamW
    'optimizer': 'Adam',
    # 训练epoch
    'epoch': 4,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 100,
    # 是否提前结束
    'is_early_stop': True,
    'patient': 2,
    'batch_size': 32,
    'max_sequence_length': 128,
    'max_sequence_len_test': 80,
    # 遗忘率
    'dropout_rate': 0.5,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'micro'
}