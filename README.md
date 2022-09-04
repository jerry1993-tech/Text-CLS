# Text-CLS

## 1、项目介绍
此 hub 是基于Tensorflow2.x的文本分类任务

通过对 Config 文件配置，可支持如下功能：

* **Bert/MacBert/RoBerta/DistilBert/AlBert/Electra/XLNet各种预训练模型训练**
* **支持二分类和多分类**   
* **支持单例测试和批量测试**
* **保存为 pb 文件可供上线部署** 
* **支持对抗训练 fgm/pgd**
* **支持 label_smoothing**

## 2、数据集
**数据：[科大讯飞中文语义病句识别挑战赛数据集](https://challenge.xfyun.cn/topic/info?type=sick-sentence-discrimination&option=stsj)** 

## 3、运行环境
* python 3.7.8
* tensorflow-gpu==2.2.0
* tensorflow-addons==0.15.0
* transformers==4.9.1
* tqdm==4.31.1
* pandas==1.3.5
* scikit-learn==1.0.2

## 4、版本情况
Version     |Describe
:-------|---
v1.0.0 |初始仓库
v2.0.0 |预训练模型基本版
v2.1.0 |添加训练 tricks


## 5、使用
### 配置文件
在config.py中配置好各个参数，文件中有详细参数说明

### 训练
参数配置完后开始模型训练
```
# [train_classifier, predict_single, predict_test, save_pb_model]
mode = 'train_classifier'
```

### 测试
训练好模型直接可以开始测试，支持单例测试和批量测试 
* 单例测试
```
# [train_classifier, predict_single, predict_test, save_pb_model]
mode = 'predict_single'
```

* 批量测试   
```
# [train_classifier, predict_single, predict_test, save_pb_model]
mode = 'predict_test'
```

## 交流
  本项目作为笔者在之前工作中项目背景下的抽象出的文本分类实验demo和trick。 
  源码和数据（实验数据）已经在项目中给出。
  
  如需要更深一步的交流，请发送消息至邮箱 1812316597@qq.com，或者在 Github 上直接留言。
