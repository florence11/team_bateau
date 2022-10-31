# -*- coding: utf-8 -*-
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using MSE.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e=y-np.dot(tx, w)
    return np.mean(e**2)*0.5

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    return np.mean(np.log(1+np.exp(tx.dot(w))) - y * tx.dot(w)) 
    

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Calculate the linear regression solution by gradient descent, returns optimal weights and mse loss.    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar representing the mse.
    """
    w=initial_w
    for n_iter in range(max_iters):
        e=y-np.dot(tx, w)
        g = -tx.T@e/len(y)
        w-= gamma*g
    loss=np.mean(e**2)*0.5
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Calculate the linear regrssion solution by stochastic gradient descent, returns optimal weights and mse loss.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar representing the mse.
    """
    w=initial_w
    batch_size=1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            e=y-np.dot(minibatch_tx, w)
            g = -tx.T@e/len(minibatch_y)
            w-= gamma*g
    loss=np.mean(e**2)*0.5
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution, returns mse, and optimal weights.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w=np.linalg.solve(tx.T@tx, tx.T@y)
    mse=compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_): 
    """implement ridge regression. 
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, hyperparameter.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE loss of the ridge regression
    """
    a=tx.T@tx+lambda_*2*len(y)*np.identity(tx.shape[1])
    w=np.linalg.solve(a, tx.T@y)
    loss=compute_loss(y,tx,w) # Compute loss without regularization term
    return w, loss


def sigmoid(x):
    """apply sigmoid function on x."""
    return 1.0/(1.0+np.exp(-x))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression. 
     Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE loss of the logistic regression
     """
    w=initial_w
    for n_iter in range(max_iters):
        w-= gamma*tx.T.dot(sigmoid(tx@w)-y)/len(y)
    loss=calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 
    """implement regulated logistic regression. 
     Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, hyperparameter.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE loss of the logistic regression (without the regularization term
     """
    w=initial_w
    for n_iter in range(max_iters):
        pred = sigmoid(tx@w)
        w-= gamma*((tx.T@(pred-y))/len(y) + 2*lambda_*w)
    loss = calculate_loss(y, tx, w) #+ lambda_ * np.squeeze(w.T@w)
    return w, loss