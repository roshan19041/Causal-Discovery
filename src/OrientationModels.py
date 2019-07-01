#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:41:26 2019

@author: roshanprakash
"""
import time
import numpy as np
import tensorflow as tf
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error as mse
from joblib import Parallel, delayed
from CausalGraphicalModel import Component
from Loss import *
import multiprocessing
        
class BasicGNN(Component):
    
    """ A Basic function approximator ; Y = f(X,noise) """
    
    def __init__(self, batch_size=256, lr=0.001, num_hidden_layers=1, nh_per_layer=[128], training_epochs=1000, test_epochs=100):
        """
        Initializes the basic GNN.
        
        PARAMETERS
        ----------
        - batch_size (int, default=256) : the size of mini-batches used while training the network
        - lr (float, default=0.001) : the learning rate for this basic Generative Neural Network
        - num_hidden_layers (int, default=1) : the number of hidden layers to be used in the network
        - nh_per_layer (list, default=[128]) : the number of hidden units in each layer of the component networks 
                                               (Requires : <num_hidden_layers>=len(<nh_per_layer>))
        - training_epochs (int, default=1000) : the number of training epochs
        - test_epochs (int, default=100) : the number of passes of data into the trained model to compute the score for a causal direction
        
        RETURNS
        -------
        - None
        """
        #tf.reset_default_graph()
        DIMENSIONS = [1, num_hidden_layers, nh_per_layer, 1]
        super(BasicGNN, self).__init__(DIMENSIONS)
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.training_epochs = training_epochs
        self.test_epochs = test_epochs
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.noise_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.observed_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.model_input = tf.concat([self.X, self.noise_inputs], axis=1)
        self.generated_data = self.forward_pass(self.model_input)
        self.loss = compute_loss(self.generated_data, self.observed_data)
        self.train_step = self.optimizer.minimize(self.loss)
        
    def run(self, input_data, ground_truth_data, sess, is_training=False):
        """ 
        Computes a forward pass through the network ; (equivalent to generating a new data instance from noise.)
        
        PARAMETERS
        ----------
        - input_data (numpy array) : the input data, shape (N, 1)
        - ground_truth_data (numpy array) : the ground truth data, shape (N, 1)
        - sess : a tensorflow session
        - is_training (bool, default=False) : if True, updates weights if necessary
        
        RETURNS
        -------
        - a numpy array of generated data and the MMD loss.
        """
        noise = np.random.normal(3, 1, ground_truth_data.shape[0])
        noise = noise[:, np.newaxis]
        if is_training:
            generated_data, loss, _ = sess.run([self.generated_data, self.loss, self.train_step], feed_dict={ \
                                                self.X:input_data, self.observed_data:ground_truth_data, self.noise_inputs:noise})
        else:
            generated_data, loss = sess.run([self.generated_data, self.loss], feed_dict={self.X:input_data, \
                                             self.observed_data:ground_truth_data, self.noise_inputs:noise})
        return generated_data, loss
        
    def _sample_batches(self, data):
        """ 
        Samples batches of training data for an epoch.
        
        PARAMETERS
        ----------
        - data (numpy array) : the input data of shape (N, 2)
        
        RETURNS
        -------
        - a list containing batches wherein each batch in a numpy array of shape [<batch_size>, d] where d is the dimensions of the data.
        """
        shuffled = np.random.permutation(data)
        return [shuffled[k*self.batch_size:k*self.batch_size+self.batch_size] for k in range(data.shape[0]//self.batch_size)]
    
    def compute_score(self, device_manager, data, direction='XY'):
        """
        Trains the Causal Generative Network, evaluates it and saves the trained model.
        
        PARAMETERS
        ----------
        - data (numpy array) : the input data of shape (N, 2)
        
        RETURNS
        -------
        - the mean test loss after training the model.
        """
        if direction=='XY':
            x_idx = 0
            y_idx = 1
        elif direction=='YX':
            x_idx = 1
            y_idx = 0
        else:
            raise ValueError('Invalid value for argument `direction`!')
        
        if device_manager.GPUs>0:
            device_count = {'GPU':device_manager.GPUs}
        else:
            device_count = {'CPU':device_manager.njobs}
            
        with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            test_loss = 0.0
            for epoch in range(1, self.training_epochs+self.test_epochs+1):
                if epoch<=self.training_epochs:
                    mini_batches = self._sample_batches(data)
                    for iteration in range(1, len(mini_batches)+1):
                        
                        generated_data, loss = self.run(mini_batches[iteration-1][:, x_idx][:, np.newaxis], \
                                                        mini_batches[iteration-1][:, y_idx][:, np.newaxis], \
                                                        sess, is_training=True)
                else:
                    shuffled_idxs = np.random.permutation(data.shape[0])
                    generated_data, loss = self.run(data[:, x_idx][:, np.newaxis][shuffled_idxs], \
                                                    data[:, y_idx][:, np.newaxis][shuffled_idxs], \
                                                    sess, is_training=False)
                    test_loss+=loss
        return test_loss/self.test_epochs
               
class OrientationNet:   
    
    """ A Pairwise GNN used for inferring causal relationship between two nodes. """
    
    def __init__(self, batch_size=256, lr=0.001, num_hidden_layers=1, nh_per_layer=[128], training_epochs=1000, \
                 test_epochs=100, max_iterations=3, runs_per_iteration=5, threshold=0.01):
        """
        Initializes orientation network.
        
        PARAMETERS
        ----------
        - batch_size (int, default=256) : the size of mini-batches used while training the network
        - lr (float, default=0.001) : the learning rate for this basic Generative Neural Network
        - num_hidden_layers (int, default=1) : the number of hidden layers to be used in the network
        - nh_per_layer (list, default=[128]) : the number of hidden units in each layer of the component networks (Requires : <num_hidden_layers>=len(<nh_per_layer>))
        - training_epochs (int, default=1000) : the number of training epochs
        - test_epochs (int, default=100) : the number of passes of data into the trained model to compute the score for a causal direction
        - max_iterations (int, default=3) : the maximum number of iterations for each direction's network 
        - runs_per_iteration (int, default=5) : the number of runs in each iteration for scoring each direction's network ;
                                                averaged results for unbiased estimates of scores
        - threshold (float, default=0.01) : the threshold for the p-value in the t-test
        
        RETURNS
        -------
        - None
        """
        self.batch_size = batch_size
        self.learning_rate = lr
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer = nh_per_layer
        self.training_epochs = training_epochs
        self.test_epochs = test_epochs
        # TTest criterion specific initializations
        self.test_threshold = threshold
        self.pval = np.inf
        self.run_count = 0 # initialize number of runs 
        self.runs_per_iteration = runs_per_iteration
        self.max_iterations = max_iterations
        self.XY_scores = []
        self.YX_scores = []
        
    def reset(self):
        """ Resets some testing characteristics """
        self.pval = np.inf
        self.run_count = 0 # re-initialize number of runs 
        self.XY_scores = []
        self.YX_scores = []
         
    def _check_stop_loop(self):
        """
        Checks if the loop for scoring direction should stop.
        
        PARAMETERS
        ----------
        - None
        
        RETURNS
        -------
        - True or False.
        """
        if self.run_count==0:
            return False
        t_statistic, self.pval = ttest_ind(self.XY_scores, self.YX_scores, equal_var=False)
        if self.run_count<self.runs_per_iteration*self.max_iterations and self.pval>self.test_threshold:
            return False
        else:
            return True
        
    def _compute_direction_score(self, data): # PARALLELIZE FUNCTION!
        """
        Computes the scores for both directions, X-->Y and Y-->X, based on a t-test between results from fitting a Basic-GNN to pairwise data, multiple times.
        
        PARAMETERS
        ----------
        - data (numpy array) : the input data of shape (N, 2)
        
        RETURNS
        -------
        - a score between -1 and 1 ; if score<0, Y-->X else X-->Y.
        """
        # setup device manager 
        device_manager = DeviceManager(autoset=True)
        while self._check_stop_loop() is False:
            for run in range(self.runs_per_iteration):
                # compute model scores for X-->Y
                tf.reset_default_graph()
                GNN_XY = BasicGNN(batch_size=self.batch_size, lr=self.learning_rate, num_hidden_layers=self.num_hidden_layers, \
                                  nh_per_layer=self.units_per_layer, training_epochs=self.training_epochs, test_epochs=self.test_epochs)
                self.XY_scores.append(GNN_XY.compute_score(device_manager, data, direction='XY'))
                # compute model scores for Y-->X
                tf.reset_default_graph()
                GNN_YX = BasicGNN(batch_size=self.batch_size, lr=self.learning_rate, num_hidden_layers=self.num_hidden_layers, \
                                  nh_per_layer=self.units_per_layer, training_epochs=self.training_epochs, test_epochs=self.test_epochs)
                self.YX_scores.append(GNN_YX.compute_score( device_manager, data, direction='YX'))
            self.run_count+=self.runs_per_iteration
        XY_score = np.mean(self.XY_scores)
        YX_score = np.mean(self.YX_scores)
        return (YX_score-XY_score)/(YX_score+XY_score)  
           
class OrientationTree:
    
    """ Decision Tree Regression Model for orienting edges. """
    
    def __init__(self, test_size=0.25):
        """
        Initialize the tree based regressor.
        
        PARAMETERS
        ----------
        - test_size (float, default=0.25) : the proportion of samples to be used as test data
        
        RETURNS
        -------
        - None
        """
        self.test_size = test_size
        
    def _fit_score(self, x, y, model, seed):
        """
        Fits a decision tree regressor to the data, for y=f(x)
        
        PARAMETERS
        ----------
        - x (numpy array) : the covariate(s), of shape (N, D) ; expected shape --> (N, 1)
        - y (numpy array) : the target(s), of shape (N, D) ; expected shape --> (N, 1)
        - model (sklearn.tree.DecisionTreeRegressor) : the model to fit to the data
        - run_idx (int) : the seed for the numpy.random ; used for reproducability
        
        RETURNS
        -------
        - the MSE on the test data, after training the model.
        """
        state = np.random.get_state()
        np.random.seed(int(0.78*seed))
        shuffled_idxs = np.random.permutation(x.shape[0])
        np.random.set_state(state)
        model.fit(x[shuffled_idxs][int(x.shape[0]*self.test_size):], y[shuffled_idxs][int(y.shape[0]*self.test_size):])
        return mse(model.predict(x[shuffled_idxs][:int(x.shape[0]*self.test_size)]), y[shuffled_idxs][:int(y.shape[0]*self.test_size)])
        
    def _compute_direction_score(self, data, nruns=16):
        """
        Fits a decision tree regressor to the data, for y=f(x)
        
        PARAMETERS
        ----------
        - data (numpy array) : the input data, of shape (N, 2) 
        - nruns (int) : the number of runs for scoring each model
        
        RETURNS
        -------
        - a score between -1 and 1 ; if score<0, Y-->X else X-->Y.
        """
        x, y = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]
        param_grid = {'max_features':['log2', 'sqrt', 0.5], \
                      'min_samples_split':[2, 8, 64, 512, 1e-2, .2, .4]}
        # y = f(x)
        cv_splits = ShuffleSplit(n_splits=3, test_size=0.2)
        model0 = DT(**GridSearchCV(DT(random_state=1), param_grid=param_grid, n_jobs=-1, cv=cv_splits).fit(x, scale(y)).best_params_)
        time.sleep(1)
        model0_scores = Parallel(n_jobs=-1)(delayed(self._fit_score)(x, scale(y), model0, _) for _ in range(nruns))
        # x = f(y)
        cv_splits = ShuffleSplit(n_splits=3, test_size=0.2)
        model1 = DT(**GridSearchCV(DT(random_state=300), param_grid=param_grid, n_jobs=-1, cv=cv_splits).fit(y, scale(x)).best_params_)
        model1_scores = Parallel(n_jobs=-1)(delayed(self._fit_score)(y, scale(x), model1, _) for _ in range(nruns))
        # direction score
        a = np.array(model0_scores)-np.array(model1_scores)
        return np.mean(a)
    
class DeviceManager:
    
    """ A device context manager. """
         
    def __init__(self, autoset=True):
        """ 
        Initialize the manager
        
        PARAMETERS
        ----------
        autoset (bool, default=True) : Looks for the system's GPU and CPU capabilities and sets up worker characteristics automatically.
        
        RETURNS
        -------
        - None.
        """
        # default characteristics
        self.njobs = multiprocessing.cpu_count()
        self.GPUs = 0
        if autoset:
            self.autoset()
            
    def autoset(self):
        """ Looks for the system's GPU and CPU capabilities and sets up worker characteristics automatically. """
        try:
            # look for ID's of user-set GPUs
            devices = ast.literal_eval(os.environ['CUDA_VISIBLE_DEVICES']) # look for CUDA supported GPU
            if type(devices)!=list and type(devices)!=tuple:
                devices = [devices]
            self.njobs = len(devices)
            self.GPUs = len(devices)
            print('Detected {} CUDA supported device(s)!'.format(self.NJOBS)) 
        except: # key error ; no environment variable called 'CUDA_VISIBLE_DEVICES'
            self.GPUs = len(GPUtil.getAvailable(order='first', limit=8, maxLoad=0.5,\
                                               maxMemory=0.5, includeNan=False))
            if self.GPUs==0:
                print('No GPU devices found! Setting n_jobs to number of CPUs..')
                self.njobs = multiprocessing.cpu_count()
            else:
                print('GPU devices found! Setting n_jobs to number of available GPU devices ..')
                self.njobs = self.GPUs