import implementations as impl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(y, tx, ratio, seed=1):
    """split the dataset based on the split ratio. """
    tr = int(np.floor(y.shape[0]*ratio))
    y_tr, y_te = y[:tr], y[tr:]
    tx_tr, tx_te = tx[:tr], tx[tr:]
    return y_tr, y_te, tx_tr, tx_te

#  LAMBDAS
def best_lambda_selection_logistic(y, tx, max_iters, gamma):
    """finds the lambda of regularized logistic regression with the smallest test loss on a split dataset.    """
    seed = 6
    lambdas = np.logspace(-20, 1, 22)

    # Split data for training for testing 
    y_tr, y_te, tx_tr, tx_te = split_data(y, tx, 0.75, seed)
    
    initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])
    losses_training = []
    losses_testing = []

    for lambda_ in lambdas:
        # print(f"Current lambda={lambda_}")
        w, loss_training = impl.reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
        loss_test=impl.calculate_loss(y_te,tx_te,w)
        losses_training.append(loss_training)
        losses_testing.append(loss_test)
        # print(f"training_loss = {loss_training}, test_loss = {loss_test}")

    ind_best_lambda = np.argmin(losses_testing)
    best_lambda = lambdas[ind_best_lambda]
    print(f"Best lambda = {best_lambda}, training_loss = {losses_training[ind_best_lambda]}, test_loss = {losses_training[ind_best_lambda]}")
    return best_lambda


def best_lambda_selection_ridge(y, tx, max_iters):
    """finds the lambda of ridge regression with the smallest test loss on a split dataset.    """
    seed = 6
    lambdas = np.logspace(-20, 1, 22)

    # Split data for training for testing 
    y_tr, y_te, tx_tr, tx_te = split_data(y, tx, 0.75, seed)

    initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])
    losses_training = []
    losses_testing = []

    for lambda_ in lambdas:
        # print(f"Current lambda={lambda_}")
        w, loss_training = impl.ridge_regression(y_tr, tx_tr, lambda_)
        loss_test = impl.compute_loss(y_te, tx_te, w)
        losses_training.append(loss_training)
        losses_testing.append(loss_test)
        # print(f"training_loss = {loss_training}, test_loss = {loss_test}")

    ind_best_lambda = np.argmin(losses_testing)
    best_lambda = lambdas[ind_best_lambda]
    print(f"Best lambda = {best_lambda}, training_loss = {losses_training[ind_best_lambda]}, test_loss = {losses_training[ind_best_lambda]}")
    return best_lambda



# GAMMAS
def best_gamma_selection(y, tx, max_iters):
    """finds the gamma of logistic regression with the smallest test loss on a split dataset.    """
    seed = 4
    gammas = np.logspace(-8, -2, 11)
    y_tr, y_te, tx_tr, tx_te = split_data(y, tx, 0.5, seed)
                              
    initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])
    losses_training = []
    losses_testing = []

    for gamma in gammas:
        # print(f"Current gamma={gamma}")
        w, loss_training = impl.logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma=gamma)
        loss_test = impl.calculate_loss(y_te,tx_te,w)
        losses_training.append(loss_training)
        losses_testing.append(loss_test)
        # print(f"training_loss = {loss_training}, testing_loss = {loss_test}")

    ind_best_gamma = np.argmin(losses_testing)
    best_gamma = gammas[ind_best_gamma]
    print(f"Best gamma = {best_gamma}, training_loss = {losses_training[ind_best_gamma]}, test_loss = {losses_testing[ind_best_gamma]}")
    return best_gamma

                        
def best_gamma_selection_gd(y, tx, max_iters):
    """finds the gamma of gradient descent linear regression with the smallest test loss on a split dataset.    """
    seed = 4
    gammas = np.logspace(-8, -2, 11)
    y_tr, y_te, tx_tr, tx_te = split_data(y, tx, 0.5, seed)
                              
    initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])
    losses_training = []
    losses_testing = []

    for gamma in gammas:
        # print(f"Current gamma={gamma}")
        w, loss_training = impl.mean_squared_error_gd(y_tr, tx_tr, initial_w, max_iters, gamma=gamma)
        loss_test = impl.compute_loss(y_te, tx_te, w)
        losses_training.append(loss_training)
        losses_testing.append(loss_test)
        # print(f"training_loss = {loss_training}, testing_loss = {loss_test}")

    ind_best_gamma = np.argmin(losses_testing)
    best_gamma = gammas[ind_best_gamma]
    print(f"Best gamma = {best_gamma}, training_loss = {losses_training[ind_best_gamma]}, test_loss = {losses_testing[ind_best_gamma]}")
    return best_gamma
      
    
def best_gamma_selection_sgd(y, tx, max_iters):
    """finds the gamma of stochastic gradient descent linear regression with the smallest test loss on a split dataset.    """
    seed = 4
    gammas = np.logspace(-8, -6, 11)
    y_tr, y_te, tx_tr, tx_te = split_data(y, tx, 0.5, seed)
                              
    initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])
    losses_training = []
    losses_testing = []

    for gamma in gammas:
        # print(f"Current gamma={gamma}")
        w, loss_training = impl.mean_squared_error_sgd(y_tr, tx_tr, initial_w, max_iters, gamma=gamma)
        loss_test = impl.compute_loss(y_te, tx_te, w)
        losses_training.append(loss_training)
        losses_testing.append(loss_test)
        # print(f"training_loss = {loss_training}, testing_loss = {loss_test}")

    ind_best_gamma = np.argmin(losses_testing)
    best_gamma = gammas[ind_best_gamma]
    print(f"Best gamma = {best_gamma}, training_loss = {losses_training[ind_best_gamma]}, test_loss = {losses_testing[ind_best_gamma]}")
    return best_gamma






# PLOTTING

def heatmap_corrolation(data):
    """finds the matrix corrolation of each column of the data set 
    (it is why we do data.T, because we don't want to compare the row, but only the column)
    plots the heatmap with the matrix of correlation between -1 and 1"""
    testcorr = np.corrcoef(data.T)
    sns.heatmap(testcorr)
    return

def boxplot_outlier_three(input_data, see_c1, see_c2, see_c3):
    """inputs=which columns we want to see.
    If we want to see column 0, column 1 and column 10 we give see_c1 = 0, see_c2 = 1 and see_c3 = 10
    displays three plots, which give us the median, the quartile and the outlier"""
    data = [input_data[:,see_c1],input_data[:,see_c2],input_data[:,see_c3]]
    plt.boxplot(data)
    plt.show()
    return

def boxplot_outlier_one(input_data, see):
    """inputs=which column we want to see.
    If we want to see column n we give see = n
    like the boxplot_outlier_three"""
    data = input_data[:,see]
    plt.boxplot(data)
    plt.show()
    return

def plot_histogram(data, col):
    """to see if there is again outliers or not, see the range of value of the column and which value is more present in the column"""
    plt.hist(data.T[col], color='gray', alpha=0.5)
    plt.title("Histogram of colum")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return
