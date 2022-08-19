# -*- coding: utf-8 -*-
# @Author :
# @File : logger.py

import logging
import datetime


def get_logger(log_dir, name):
    log_file = log_dir + '/' + (datetime.datetime.now().strftime('%Y%m%d%H%M%S-' + name + '.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')

    # load log to file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # show log on terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return logger

