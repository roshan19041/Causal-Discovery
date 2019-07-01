#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:46:10 2019

@author: roshanprakash
"""

import pandas as pd
import numpy as np
import tensorflow as tf
tf.set_random_seed(35)
np.random.seed(3)
import networkx as nx
from GraphUtils import *
from Loss import *

class Component:
    """ A function approximator for a node in the causal graph. Requires the number of parents for this node, the number of hidden layers
        to be used in the regression network, the number of units per layer and the dimension of the output.
    """
    def __init__(self, dimensions):
        """
        Initialize the function approximator for a node.
        
        PARAMETERS
        ----------
        - dimensions(list) : a list of the form [number_parents, num_hidden_layers, [number_hidden_units,...], output_dim]
        
        RETURNS
        -------
        - None
        """
        self.num_parents = dimensions[0]
        self.num_layers = dimensions[1]+1
        assert len(dimensions[2])==self.num_layers-1, "Number of hidden units must be specified for each fully-connected hidden layer!"
        self.layers = {}
        for i in range(1, self.num_layers):
            self.layers['layer_{}'.format(i)] = tf.keras.layers.Dense(dimensions[2][i-1], kernel_initializer=tf.variance_scaling_initializer, activation = tf.nn.relu)
        self.layers['layer_{}'.format(self.num_layers)] = tf.keras.layers.Dense(dimensions[-1], kernel_initializer=tf.truncated_normal_initializer) # final output layer
          
    def forward_pass(self, x):
        """
        Computes a forward pass through this node network.
        
        PARAMETERS
        ----------
        - x(tf.tensor of shape(N, 1+number_of_parents)) : the input data to this node (node_out = f(node_outs(parents(node)), noise))
        
        RETURNS
        -------
        - the node output (tf.tensor of shape (N, 1))
        """
        for l in range(1, self.num_layers+1):
            if l==1:
                out = self.layers['layer_{}'.format(l)](x)
            else:
                out = self.layers['layer_{}'.format(l)](out)
        return out
    
    def reset_weights(self):
        """ Resets the weights of the network """
        pass
    
class CausalNet:
    """ 
    A Causal Generative Network that collectively holds component-regression networks for every node in the causal graph ;
    Automatically infers a directed causal graph from observational data and initializes the network based on this DAG.
        
    REFERENCE
    ---------
    [1.] Learning Functional Causal Models with Generative Neural Networks 
    [2.] Link : https://arxiv.org/pdf/1709.05321.pdf  
    """
    def __init__(self, data, batch_size=256, lr=0.001, num_hidden_layers=1, nh_per_layer=[64]):
        """ 
        Initialize the causal generative network.
        
        PARAMETERS
        ----------
        - data(pandas DataFrame) : the input observational data
        - batch_size(int, default=256) : the size of mini-batches used while training the network
        - lr(float, default=0.001) : the learning rate for this Causal Generative Neural Network
        - num_hidden_layers(int, default=1) : the number of hidden layers to be used in each of the component networks
        - nh_per_layer(list, default=[64]) : the number of hidden units in each layer of the component networks (Requires : <num_hidden_layers>=len(<nh_per_layer>))
        
        RETURNS
        -------
        - None
        """
        # Graph-specific initializations
        self.data = data
        assert batch_size<=self.data.values.shape[0], "Not enough data instances. Reduce the batch size and try again!"
        self.batch_size = batch_size
        self.causal_graph = infer_DAG(self.data)
        self.topological_order = nx.topological_sort(self.causal_graph)
        # Network-specific initializations
        tf.reset_default_graph()
        self.Network = self._build_causal_network(num_hidden_layers, nh_per_layer)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.output_layer = num_hidden_layers+1
        self.observed_data = tf.placeholder(dtype=tf.float32, shape=[None, self.data.shape[1]])
        self.noise_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.data.shape[1]])
        self.node_inputs = {}
        self.node_outputs = {}
        # forward-pass through the collective network
        for node in self.topological_order:
            node_idx = list(self.data.columns).index(node)
            self.node_inputs[node] = tf.concat([j for i in [[self.node_outputs[parent] for parent in self.causal_graph.predecessors(node)],\
                                      [tf.slice(self.noise_inputs, begin=[0, node_idx], size=[-1, 1])]] for j in i], axis=1)
            self.node_outputs[node] = self.Network[node].forward_pass(self.node_inputs[node])
        self.generated_data = tf.concat([self.node_outputs[node] for node in self.causal_graph.nodes], axis=1)
        self.loss = compute_loss(self.generated_data, self.observed_data)
        self.train_step = self.optimizer.minimize(self.loss)
        
    def _build_causal_network(self, num_hidden_layers, nh_per_layer):
        """ 
        Helper function to build a collective network by first building networks for every node in the graph.
        
        PARAMETERS
        ----------
        - num_hidden_layers(int, default=1) : the number of hidden layers to be used in each of the component networks
        - nh_per_layer(list, default=[64]) : the number of hidden units in each layer of the component networks 
        									 (Requires : <num_hidden_layers>=len(<nh_per_layer>))
        
        RETURNS
        -------
        - a dictionary of the form {node_id : Component-Network-object} ; all nodes included.
        """
        collective = {}
        for node in self.causal_graph.nodes:
            collective[node] = Component([len(list(self.causal_graph.predecessors(node))), num_hidden_layers, nh_per_layer, 1])
        return collective
        
    def run(self, ground_truth_data, sess, is_training=False):
        """ 
        Computes a forward pass through the network ; (equivalent to generating a new data instance from noise.)
        
        PARAMETERS
        ----------
        - ground_truth_data(numpy array) : the ground truth data
        - sess : a tensorflow session
        - is_training(bool, default=False) : if True, updates weights if necessary
        
        RETURNS
        -------
        - a numpy array of generated data and the MMD loss.
        """
        noise = np.hstack([np.reshape(np.random.normal(3, 1, ground_truth_data.shape[0]), (ground_truth_data.shape[0], 1)) for i in range(self.data.shape[1])])
        if is_training:
            generated_data, loss, _ = sess.run([self.generated_data, self.loss, self.train_step], feed_dict={self.observed_data:ground_truth_data, self.noise_inputs:noise})
            print('Updated weights')
        else:
            generated_data, loss = sess.run([self.generated_data, self.loss], feed_dict={self.observed_data:ground_truth_data, self.noise_inputs:noise})
        return generated_data, loss
    
    def _sample_batches(self):
        """ 
        Samples batches of training data for an epoch.
        
        PARAMETERS
        ----------
        - None
        
        RETURNS
        -------
        - a list containing batches wherein each batch in a numpy array of shape [<batch_size>, d] where d is the dimensions of the data.
        """
        shuffled = np.random.permutation(self.data.values[:int(0.75*self.data.shape[0])])
        return [shuffled[k*self.batch_size:k*self.batch_size+self.batch_size] for k in range(int(0.75*self.data.shape[0])//self.batch_size)]
    
    def train_evaluate(self, num_epochs=100, print_every=1000, plot_losses=False, save_path='../model'):
        """
        Trains the Causal Generative Network, evaluates it and saves the trained model.
        
        PARAMETERS
        ----------
        - num_epochs(int, default=100 : the number of training epochs
        - print_every(int, default=1000) : Prints loss after every <print_every> iterations
        
        RETURNS
        -------
        - None ; plots the training curve and saves the model to <save_path>.
        """
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            for epoch in range(1, num_epochs+1):
                mini_batches = self._sample_batches()
                for iteration in range(1, len(mini_batches)+1):
                    generated_data, loss = self.run(mini_batches[iteration-1], sess, is_training=True)
                    if iteration%print_every==0:
                        print('Completed training step for iteration {}, loss = {}'.format(iteration*epoch, loss))
                    training_losses.append(loss)
            print(generated_data)
            print('Completed training the model!')
            if plot_losses:
                # plot losses here
                pass
            # Model evaluation
            print('Validating model..')
            _, test_loss = self.forward_pass(self.data[int(0.75*self.data.shape[0]):], sess, is_training=False)
            print('Mean test-time loss = {}'.format(test_loss))
            # save model here
            
if __name__=='__main__':
    sample_data = 1000*np.random.rand(100, 4)
    df = pd.DataFrame(sample_data, columns = np.arange(2, 6))
    CGNN = CausalNet(df, lr=0.0001, batch_size=10, num_hidden_layers=3, nh_per_layer=[100, 80, 20])
    CGNN.train_evaluate()