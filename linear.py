import numpy as np
import cv2, os, inspect, re, math
import sklearn as sk

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from read import *

###### Parameters
f_in_trn = 'Data/train_28.csv'
f_in_tst = 'Data/test_28.csv'
sol_dir = 'Data/train_solutions.csv'
my_lam = 5
###### 

# Take care of some file i/o
my_file = re.search(r'_[0-9]+', f_in_trn).group()
my_dim = re.search(r'[0-9]+', my_file).group()
file_name = inspect.getfile(inspect.currentframe())
f_out = 'Submissions/ks_' + str(my_dim) + '.csv'

def read_X_Y():
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


def split_data(nrows, train = .6, val = .2, test = .2):
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
        for index, lam_i in enumerate(lam_range):
            model = ridge_train(Xtrn_trn, Ytrn_trn, lam = lam_i)
            Ytrn_val_pred = ridge_pred(model, Xtrn_val)
            #val_err[i, index] = rmse(Ytrn_val[::, 1:], Ytrn_val_pred)
            val_err[i, index] = mean_squared_error(Ytrn_val[::, 1:], Ytrn_val_pred)
    colmeans = np.mean(val_err, axis=0)
    lam_star = lam_range[np.argmin(colmeans)]
    print(file_name + ': Best lambda was found to be: ' + str(lam_star))
    return val_err


def output_Ytst(Ytst, save_csv=True, return_Y=False):
    """
    Get Ytst ready for kaggle output, save CSV
    """
    # Fix up test response 
    print(file_name + ': Forcing [0,1] bounds')
    Ytst = force_bounds(Ytst)
    print(file_name + ': Adding ID column to response')
    Ytst_names = get_image_names(tst_dir)
    Ytst = np.c_[Ytst_names, Ytst]

    if (save_csv):
        # Output submission
        print(file_name + ': Saving csv to ' + f_out)
        colfmt = ['%i'] + ['%f'] * (Ytst.shape[1] - 1)
        np.savetxt(f_out, Ytst, delimiter = ',', fmt = colfmt)

    if (return_Y):
        return Y
    else:
        return 


def main():
    """
    Run experiment
    """
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y()
    val_err = ridge_cv(Xtrn[0:1000], Ytrn[0:1000], lam_range = np.arange(3), n_trials = 10)
    #output_Ytst(Ytst)
    return 

if __name__ == "__main__":
    main()
