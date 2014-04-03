import numpy as np
import cv2, os, inspect, re, math, datetime
import sklearn as sk

import matplotlib.pyplot as plt
import pandas as pd
import pandas.rpy.common as com
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from read import *

###### Parameters
f_in_trn = 'Data/train_28.csv'
f_in_tst = 'Data/test_28.csv'
sol_dir = 'Data/train_solutions.csv'
my_lam = 10
###### 

# Take care of some file i/o
my_file = re.search(r'_[0-9]+', f_in_trn).group()
my_dim = re.search(r'[0-9]+', my_file).group()
file_name = inspect.getfile(inspect.currentframe())
# Change the f_out!
f_out = 'Submissions/ls_clr_' + str(my_dim) + '.csv'

def read_X_Y(f_in_trn, f_in_tst, sol_dir, my_dim):
    """
    Read in pre-processed matricies & train solutions
    """
    print(file_name + ': Reading train/test matrix w/ dim = ' + str(my_dim))
    Xtrn = ensure_dim(np.loadtxt(open(f_in_trn, 'rb'), delimiter = ',', skiprows = 0))
    Xtst = ensure_dim(np.loadtxt(open(f_in_tst, 'rb'), delimiter = ',', skiprows = 0))
    print(file_name + ': Reading training solutions')
    Ytrn = np.loadtxt(open(sol_dir, 'rb'), delimiter = ',', skiprows = 1)
    return (Xtrn, Xtst, Ytrn, my_dim)


def ls_train(Xtrn, Ytrn):
    """
    Train for least squares
    """
    print(file_name + ': Training [least squares] regression')
    model = sk.linear_model.LinearRegression()
    model.fit(Xtrn, Ytrn[::, 1:])
    return model

def ls_pred(model, Xtst):
    """
    Predict on new matrix for least squares
    """
    print(file_name + ': Predicting on new matrix')
    Ytst = model.predict(Xtst)
    return Ytst


def rmse(y, y_pred, ax=None):
    err = np.sqrt(((y - y_pred) ** 2).mean(axis=ax))
    return err

def split_data(nrows, train = .6, val = .35, test = .05):
    """
    Randomly partition data into train/val/test indicies
    """
    # Sample train: if odd, take ceiling --> more training obs
    rows = np.arange(nrows)
    trainInd = np.random.choice(rows, size=int(np.ceil(nrows*train)), replace=False)
    # Sample val: first remove training rows, take floor --> less val obs
    rows = np.setdiff1d(rows, trainInd)
    valInd = np.random.choice(rows, size=int(np.floor(nrows*val)), replace=False)
    # Sample test: just take rows that aren't in train/val
    rows = np.setdiff1d(rows, valInd)
    testInd = rows
    return (trainInd, valInd, testInd)


def ridge_train(Xtrn, Ytrn, lam):
    """
    Train for ridge
    """
    print(file_name + ': Training [ridge] regression w/ reg = ' + str(lam))
    model = sk.linear_model.Ridge(alpha = lam, solver = 'svd')
    #model = sk.linear_model.Ridge(alpha = lam)
    model.fit(Xtrn, Ytrn[::, 1:])
    #print('\t first model coefficient: ' + str(model.coef_[1]))
    return model

def ridge_pred(model, Xtst):
    """
    Predict on new matrix for ridge
    """
    print(file_name + ': \t Predicting on new matrix')
    Ytst = model.predict(Xtst)
    return Ytst

def ridge_cv(Xtrn, Ytrn, lam_range = np.arange(0, 10.5, 0.5), n_trials = 10):
    """
    Cross-validation to find regularization param
    """
    print(file_name + ': Cross-validating for reg. param.')
    val_err = np.zeros((n_trials, len(lam_range)))
    for i in xrange(n_trials):
        (trnInd, valInd, tstInd) = split_data(Xtrn.shape[0])
        Xtrn_trn = Xtrn[trnInd]; Ytrn_trn = Ytrn[trnInd]
        Xtrn_val = Xtrn[valInd]; Ytrn_val = Ytrn[valInd]
        Xtrn_tst = Xtrn[tstInd]; Ytrn_tst = Ytrn[tstInd]
        Xtrn_trn = preprocessing.scale(Xtrn_trn)
        Xtrn_tst = preprocessing.scale(Xtrn_tst)
        for index, lam_i in enumerate(lam_range):
            model = ridge_train(Xtrn_trn, Ytrn_trn, lam = lam_i)
            Ytrn_val_pred = ridge_pred(model, Xtrn_val)
            #val_err[i, index] = np.sqrt(mean_squared_error(Ytrn_val[::, 1:], Ytrn_val_pred))
            val_err[i, index] = rmse(Ytrn_val[::, 1:], Ytrn_val_pred)
    colmeans = np.mean(val_err, axis=0)
    lam_star = lam_range[np.argmin(colmeans)]
    print(file_name + ': Best lambda was found to be: ' + str(lam_star))
    return (lam_star, val_err)

def ridge_cv_plot(val_err, lam_range):
    """
    Source: http://rpy.sourceforge.net/rpy2/doc-2.3/html/graphics.html
    """
    base = importr('base')
    df = pd.DataFrame(val_err, columns = lam_range)
    df = pd.melt(df)
    df_r = com.convert_to_r_dataframe(df)
    # Create boxplot
    gp = ggplot2.ggplot(df_r)
    pp = gp + \
         ggplot2.aes_string(x='factor(variable)', y='value') + \
         ggplot2.geom_boxplot() + \
         ggplot2.ggtitle("Validation Error by Lambda")
    pp.plot()
    return

def learning_curve(Xtrn, Ytrn, method='linear', lam_range=[0,0.1,1,10,100]):
    """
    Plot learning curve for a given method
    """
    # Split data, subset data, set model
    (trnInd, valInd, tstInd) = split_data(Xtrn.shape[0], .70, 0, .30)
    Xtrn_trn = Xtrn[trnInd]; Ytrn_trn = Ytrn[trnInd]
    Xtrn_tst = Xtrn[tstInd]; Ytrn_tst = Ytrn[tstInd]
    model = sk.linear_model.LinearRegression()
    # Calculate rmse for each number of obs
    obs_range = range(10, len(Xtrn_trn), 2500)
    trn_err = []; tst_err = []
    for i in obs_range:
        if ((i-10) % 10000) == 0: print '\t training example: %i' % i
        if method=='linear':
            model.fit(Xtrn_trn[0:i,::], Ytrn_trn[0:i, 1:])
            Ytrn_trn_pred = model.predict(Xtrn_trn[0:i,::])
            Ytrn_tst_pred = model.predict(Xtrn_tst[0:i,::])
            trn_err.append(rmse(Ytrn_trn[0:i, 1:], Ytrn_trn_pred))
            tst_err.append(rmse(Ytrn_tst[0:i, 1:], Ytrn_tst_pred))
        if method=='ridge':
            # Use validation set to find best lambda for a given #of obs
            #print "%i %i" % (len(Xtrn[0:i,::]),len(Ytrn[0:i,1:]))
            (lam_star, val_err) = ridge_cv(Xtrn[0:i,::], Ytrn[0:i,1:], lam_range, n_trials = 5)
            # Train model using lambda star
            model = ridge_train(Xtrn[0:i,::], Ytrn_trn[0:i,::], lam_star)
            Ytrn_trn_pred = ridge_pred(model, Xtrn_trn[0:i,::])
            Ytrn_tst_pred = ridge_pred(model, Xtrn_tst[0:i,::])
            trn_err.append(rmse(Ytrn_trn[0:i, 1:], Ytrn_trn_pred))
            tst_err.append(rmse(Ytrn_tst[0:i, 1:], Ytrn_tst_pred))
    # Plot curve
    plt.plot(obs_range, trn_err)
    plt.plot(obs_range, tst_err)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return

def bias_var():
    """
    Plot bias/var tradeoff
    """
    return 


def output_Ytst(Ytst, tst_dir, f_out, save_csv=True, return_Y=False):
    """
    Get Ytst ready for kaggle output, save CSV
    """
    # Fix up test response 
    print(file_name + ': Forcing [0,1] bounds')
    Ytst = force_bounds(Ytst)
    print(file_name + ': Adding ID column to response')
    Ytst_names = get_image_names(tst_dir)
    Ytst = np.c_[Ytst_names, Ytst]
    # Output submission [if desired]
    if (save_csv):
        print(file_name + ': Saving csv to ' + f_out)
        colfmt = ['%i'] + ['%f'] * (Ytst.shape[1] - 1)
        np.savetxt(f_out, Ytst, delimiter = ',', fmt = colfmt)
    # Return Ytst matrix [if desired]
    if (return_Y):
        return Ytst
    return 


def svm_regression(Xtrn, Xtst, Ytrn, lam):
    from sklearn.svm import SVR
    model = SVR(kernel = 'linear', C=1/lam)
    model.fit(Xtrn, Ytrn[::, 1:])
    Ytst = model.predict(Xtst)
    return Ytst

def main():
    """
    Run experiment
    """
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y(f_in_trn, f_in_tst, sol_dir, my_dim)
     
    #lam_range = np.arange(0, 105, 5)
    #lam_range = np.array([1,100,500,1000, 5000, 10000, 100000, 1000000])
    #lam_range = np.array([0, 0.2, 0.5, 1, 10, 100])
    #(lam_star, val_err) = ridge_cv(Xtrn[0:5000], Ytrn[0:5000], lam_range, n_trials = 10)
    #ridge_cv_plot(val_err, lam_range)
    #clf = sk.linear_model.RidgeCV(alphas = lam_range)
    #clf.fit(Xtrn, Ytrn[::, 1:])

    # Plot learning curve and bias/var tradeoff 
    #learning_curve(Xtrn, Ytrn, method='linear')

    #model = ridge_train(Xtrn, Ytrn, 10)
    #Ytst = ridge_pred(model, Xtst)
    #model = ls_train(Xtrn, Ytrn)
    #Ytst = ls_pred(model, Xtst)

    Ytst = svm_regression(Xtrn, Xtst, Ytrn, 10)
    output_Ytst(Ytst)
    return 

if __name__ == "__main__":
    main()
