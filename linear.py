import numpy as np
import pandas as pd
import cv2, os, inspect, re
import sklearn as sk
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
#from sklearn.ensemble import RandomForestRegressor
from read import *

###### Parameters
f_in_trn = 'Data/train_24.csv'
f_in_tst = 'Data/test_24.csv'
sol_dir = 'Data/train_solutions.csv'
#my_alphas = [0.1, 0.5, 1, 10, 20, 30, 40, 50, 100, 150, 200, 300]
#my_alpha = 1
###### 

# Take care of some file i/o
my_file = re.search(r'train_[0-9]+', f_in_trn).group()
my_dim = re.search(r'[0-9]+', my_file).group()
file_name = inspect.getfile(inspect.currentframe())
f_out = 'Submissions/knn_' + str(my_dim) + '.csv'

def read_X_Y():
    # Read in pre-processed matricies & train solutions
    print(file_name + ': Reading train/test matrix w/ dim = ' + my_dim)
    Xtrn = ensure_dim(np.loadtxt(open(f_in_trn, 'rb'), delimiter = ',', skiprows = 0))
    Xtst = ensure_dim(np.loadtxt(open(f_in_tst, 'rb'), delimiter = ',', skiprows = 0))
    print(file_name + ': Reading training solutions')
    Ytrn = np.loadtxt(open(sol_dir, 'rb'), delimiter = ',', skiprows = 1)

    return (Xtrn, Xtst, Ytrn, my_dim)

def least_squares(Xtrn, Xtst, Ytrn):
    # Train linear regression
    print(file_name + ': Training [least squares] regression')
    model = sk.linear_model.LinearRegression()
    model.fit(Xtrn, Ytrn[::, 1:])
    
    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst

def knn(Xtrn, Xtst, Ytrn):
    # Train linear regression
    print(file_name + ': Training [knn] regression')
    model = sk.neighbors.KNeighborsRegressor()
    model.fit(Xtrn, Ytrn[::, 1:])
    
    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst
    

def svm(Xtrn, Xtst, Ytrn):
    """
    # Train svm regression
    print(file_name + ': Training [svm] regression')
    model = sk.svm.SVR(kernel = 'linear')
    model.fit(Xtrn, Ytrn[::, 1:])

    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst
    """

def ridge(Xtrn, Xtst, Ytrn):
    '''
    # Cross-validation to find regularization parameter
    print(file_name + ': Cross-validating for reg. param.')
    CV = sk.linear_model.LassoCV(alphas = my_alphas, cv = None)
    CV.fit(Xtrn, Ytrn[::, 1:])
    print(file_name + ': Best lambda was found to be: ' + str(CV.alpha_))

    print(file_name + ': Training [ridge] regression w/ reg = ' + str(my_alpha))
    model = sk.linear_model.Lasso(alpha = my_alpha)
    model.fit(Xtrn, Ytrn[::, 1:])
    print(model.coef_[1])

    #model = RandomForestRegressor()
    #model.fit(Xtrn, Ytrn[::, 1:])
    
    # Predict on text matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst
    '''
    

def output_Ytst(Ytst):
    # Fix up test response 
    print(file_name + ': Forcing [0,1] bounds')
    Ytst = force_bounds(Ytst)
    print(file_name + ': Adding ID column to response')
    Ytst_names = get_image_names(tst_dir)
    Ytst = np.c_[Ytst_names, Ytst]
    
    # Output submission
    print(file_name + ': Saving csv to ' + f_out)
    colfmt = ['%i'] + ['%f'] * (Ytst.shape[1] - 1)
    np.savetxt(f_out, Ytst, delimiter = ',', fmt = colfmt)


def main():
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y()
    #Ytst = least_squares(Xtrn, Xtst, Ytrn)
    Ytst = knn(Xtrn, Xtst, Ytrn)
    output_Ytst(Ytst)

if __name__ == "__main__":
    main()
