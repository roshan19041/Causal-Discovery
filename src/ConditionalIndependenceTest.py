"""
Code borrowed/reproduced from kjchalup's 'A fast conditional independence test'

Reference: Chalupka, Krzysztof and Perona, Pietro and Eberhardt, Frederick, 2017.

@author: roshanprakash
"""
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import time
from scipy.stats import ttest_1samp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.model_selection import GridSearchCV
        
def _mix_merge_columns(x, z, seed=None):
    """
    Permutes the columns of two samples separately and merges them.
    
    PARAMETERS
    ----------
    - x (numpy array) : the first set of random variables, of shape (N, D1)
    - y (numpy array) : the next set of random variables, of shape (N, D2)
    
    RETURNS
    -------
    - a numpy array of shape (N, D1+D2), containing permuted columns.      
    """
    num_columns = x.shape[1]+z.shape[1]
    global_state = np.random.get_state() 
    np.random.seed(seed or int(time.time())) 
    shuffled_idxs = np.random.permutation(np.arange(num_columns))
    np.random.set_state(global_state)  # set the global state back to what it was
    reordered_out = np.zeros([x.shape[0], num_columns])
    reordered_out[:, shuffled_idxs[:x.shape[1]]] = x
    reordered_out[:, shuffled_idxs[x.shape[1]:]] = z
    return reordered_out
    
def _find_best_model(x, y, z, params_grid, test_size, log_features=False):
    """
    Performs GridSearch on `params_grid`.
    
    PARAMETERS
    ----------
    - x (numpy array) : the input set of random variables, of shape (N, D1)
    - y (numpy array) : the target set of random variables, of shape (N, D2)
    - z (numpy array) : the conditioning set of random variables, of shape (N, D3)
    - params_grid (dict) : the hyperparameters to try out while performing grid search ; for more details,
                           look up `sklearn.model_selection.GridSearchCV`
    - test_size (float) : the proportion of samples to be used as test data 
    - log_features (bool, default=False) : if True 'log2' will be used as `max_features` for the Decision Tree
                                           Regressor provided there are atleast 10 features in the input
    
    RETURNS
    -------
    - the Decision Tree Regressor with the optimal value for `min_sample_split`.
    """
    model_input = _mix_merge_columns(x, z)
    if log_features and model_input.shape>10:
        max_features = 'log2'
    else:
        max_features = 'auto'
    cv_splits = ShuffleSplit(n_splits=3, test_size=test_size)
    best_params = GridSearchCV(DT(max_features=max_features), params_grid, cv=cv_splits, n_jobs=-1).fit(model_input, y).best_params_
    best_model = DT(**best_params)
    return best_model
    
def _compute_error(data_tuple):
    """
    Fits the decision tree regression model to a data set, and computes the error on the test set.
    
    PARAMETERS
    ----------
    - data_dict (dict) : a dictionary containing the covariates, target and the decision tree model to be fitted.
    - proportion_test (float) : the fraction of samples to be included in test set 
    - i (int) : the run index used to access the shuffled indices of data for this run and the seed to shuffle columns 
                before merging `x` and `z`
    
    RETURNS
    -------
    - The model error on the test set.           
    """
    data_dict, proportion_test, i = data_tuple
    model = data_dict['model']
    n_test = data_dict['n_test']
    shuffled_idxs = data_dict['shuffled_idxs'][i]
    if data_dict['reshuffle']:
        perm_idxs = np.random.permutation(data_dict['x'].shape[0])
    else:
        perm_idxs = np.arange(data_dict['x'].shape[0])
    # mix up columns before training
    x = _mix_merge_columns(data_dict['x'][perm_idxs], data_dict['z'], i)
    model.fit(x[shuffled_idxs][n_test:], data_dict['y'][shuffled_idxs][n_test:])
    return mse(data_dict['y'][shuffled_idxs][:n_test], model.predict(x[shuffled_idxs][:n_test]))
    
def test_conditional_independence(x, y, z, nruns=8, params_grid={'min_samples_split':[2, 8, 64, 512, 1e-2, .2, .4]}, test_size=0.1, threshold=0.01):
    """
    Performs fast conditional/unconditional independence tests using Decision Tree Regressors.
    
    PARAMETERS
    ----------
    - x (numpy array) : the first set of random variables, of shape (N, D1)
    - y (numpy array) : the next set of random variables, of shape (N, D2)
    - z (numpy array) : the conditioning set of random variables, of shape (N, D3)
    - params_grid (dict) : the hyperparameters to try out while performing grid search ; for more details,
                           look up `sklearn.model_selection.GridSearchCV`
    - test_size (float, default=0.1) : the proportion of samples to be used as test data 
    - threshold (float, default=0.01) : the alpha value for t-test
    
    RETURNS
    -------
    - True, if X is conditionally independent of Y, given Z and False otherwise.               
    """
    assert x.shape[0]==y.shape[0], 'X and Y should contain the same number of data instances!'
    num_instances = x.shape[0]
    num_test_instances = int(test_size*num_instances)
    shuffled_idxs = [np.random.permutation(num_instances) for i in range(nruns)]
    y = StandardScaler().fit_transform(y)
    
    # find the best-fitting decision regression tree for y = f(x, z) and then train and compute error for each of `nruns`
    best_model = _find_best_model(x, y, z, params_grid, test_size)
    data_dict = {'x':x, 'y':y, 'z':z, 'model':best_model, 'reshuffle':False, 'shuffled_idxs':shuffled_idxs, 'n_test':num_test_instances}
    results_xz = np.array(Parallel(n_jobs=-1, max_nbytes=100e6)(delayed(_compute_error)((data_dict, test_size, run_idx)) for run_idx in range(nruns)))
    
    # find the best-fitting decision regression tree for : y = f(reshuffle(z)) if z is not empty, else y = f(reshuffle(x))
    if z.shape[1]==0:
        x_ = x[np.random.permutation(num_instances)]
    else: 
        x_ = np.empty(shape=[num_instances, 0])
    data_dict['best_model'] = _find_best_model(x, y, z, params_grid, test_size)
    data_dict['reshuffle'] = True
    data_dict['x'] = x_
    results_z = np.array(Parallel(n_jobs=-1, max_nbytes=100e6)(delayed(_compute_error)((data_dict, test_size, run_idx)) for run_idx in range(nruns)))

    # perform 1-sample t-test to check significance of both samples of results
    t_stat, p_val = ttest_1samp(results_z/results_xz, 1)
    if t_stat<0:
        p_val = 1-p_val/2
    else:
        p_val = p_val/2
        
    # return if samples are independent or otherwise
    if p_val<threshold:
        return False
    else:
        return True
 
if __name__=='__main__':
    data = np.zeros((10000, 4))
    data[:, 0] = np.random.normal(loc=10.0, scale=5.0, size=10000)
    data[:, 1] = np.random.normal(loc=1.0, scale=2.0, size=10000)
    data[:, 2] = np.random.gamma(2, 0.65, 10000)
    data[:, 3] = data[:, 1]+data[:, 2]
    data = pd.DataFrame(data)
    print(test_conditional_independence(data[0].values[:, np.newaxis], data[1].values[:, np.newaxis], data[[]].values))