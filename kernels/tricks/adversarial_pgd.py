# -*- coding: utf-8 -*-
# @Author : xuyingjie
# @File : adversarial_pgd.py

import tensorflow as tf


class PGD(object):
    __doc__ = """对抗训练 PGD"""

    def __init__(self, variables, gradients, gan_tape, loss):
        self.variables = variables
        self.gradients = gradients
        self.gan_tape = gan_tape
        self.loss = loss

    def attack_calculate(self, alpha=0.3, epsilon=1.0, k=3):
        origin_embedding = tf.Variable(self.variables[0])
        accum_vars = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in self.gradients]
        origin_gradients = [accum_vars[i].assign_add(grad) for i, grad in enumerate(self.gradients)]

        variables = self.variables
        gradients = self.gradients
        for t in range(k):
            embedding = variables[0]
            embedding_gradients = gradients[0]
            embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
            delta = alpha * embedding_gradients / tf.norm(embedding_gradients, ord=2)
            variables[0].assign_add(delta)

            r = variables[0] - origin_embedding
            if tf.norm(r, ord=2) > epsilon:
                r = epsilon * r / tf.norm(r, ord=2)
            variables[0].assign(origin_embedding + tf.Variable(r))

            if t != k - 1:
                gradients = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
            else:
                gradients = origin_gradients

            gan_gradients = self.gan_tape.gradient(self.loss, variables)
            gradients = [gradients[i].assign_add(grad) for i, grad in enumerate(gan_gradients)]
        variables[0].assign(origin_embedding)

        return variables, gradients

