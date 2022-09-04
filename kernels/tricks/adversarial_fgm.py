# -*- coding: utf-8 -*-
# @Author : xuyingjie
# @File : adversarial_fgm.py

import tensorflow as tf


class FGM(object):
    __doc__ = """对抗训练 FGM"""

    def __init__(self, variables, gradients, gan_tape, loss):
        self.variables = variables
        self.gradients = gradients
        self.gan_tape = gan_tape
        self.loss = loss

    def attack_calculate(self, epsilon=1.0):
        variables = self.variables
        embedding = self.variables[0]
        embedding_gradients = self.gradients[0]
        embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
        delta = epsilon * embedding_gradients / tf.norm(embedding_gradients, ord=2)

        accum_vars = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in self.gradients]
        gradients = [accum_vars[i].assign_add(grad) for i, grad in enumerate(self.gradients)]
        variables[0].assign_add(delta)

        gan_gradients = self.gan_tape.gradient(self.loss, variables)
        gradients = [gradients[i].assign_add(grad) for i, grad in enumerate(gan_gradients)]
        variables[0].assign_sub(delta)

        return variables, gradients


