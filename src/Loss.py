#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:59:20 2019

@author: roshanprakash
"""
import tensorflow as tf

def compute_loss(generated_data, observed_data):
    """
    Computes the Maximum Mean Discrepancy between generated data and observational data.
    
    PARAMETERS
    ----------
    - generated_data (numpy array) : the generated data of shape (N, D)
    - observed_data (numpy array) : the corresponding ground truth data of shape (N, D)
    
    RETURNS
    -------
    - the MMD loss.
    
    REFERENCE
    ---------
    [1.] Training generative neural networks via Maximum Mean Discrepancy optimization
    [2.] Link : https://arxiv.org/pdf/1505.03906.pdf
    """
    N = tf.cast(tf.shape(observed_data)[0], dtype=tf.float32)
    GAMMA = tf.constant(0.01, dtype=tf.float32, name='gamma')
    MULTIPLIERS = tf.concat([tf.ones([N, 1])/N, tf.ones([N, 1])/-N], axis=0)
    X = tf.concat(values=[generated_data, observed_data], axis=0)
    DOTS = tf.matmul(X, tf.transpose(X))
    SQUARE_SUMS = tf.transpose(tf.reduce_sum(tf.square(X), axis=1, keepdims=True))
    EXPONENT_TERMS = tf.add_n([tf.scalar_mul(-2, DOTS), tf.broadcast_to(SQUARE_SUMS, tf.shape(DOTS)), \
                              tf.broadcast_to(tf.transpose(SQUARE_SUMS), tf.shape(DOTS))])
    MMDLoss = tf.reduce_sum(tf.multiply(MULTIPLIERS, tf.exp(tf.scalar_mul(-GAMMA, EXPONENT_TERMS))))
    return MMDLoss