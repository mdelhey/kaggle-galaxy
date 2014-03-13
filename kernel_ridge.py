import os, inspect, re
import numpy as np
from read import *
from linear import *

def linear_kernel(X, c = 0, vec = False):
    print(file_name + ': \t Using linear kernel')
    if ~vec:
        #Ktrn = np.dot(Xtrn, Xtrn.T) + c
        K = np.dot(Xtrn[1:10,::], Xtrn[1:10,::].T)
    if vec:
        K = np.dot(X.T,X)
    return K
        

def rbf_kernel(Xtrn, Xtst, sigma = 1):
    print(file_name + ': \t Using radial basis kernel')
    return (Ktrn, Ktst)

def predict_kernel(Xst, a):
    Xtst_n = Xtst.shape[0]
    Ytst = np.zeros((Xtst_n, 38))
    for i in xrange(Xtst_n):
        Ktst_vec = linear_kernel(Xtst[i,::], vec = True)
        Ytst[i,::] = a.T * Ktst_vec
        if (i%5000==0): print(str(i))
    return

def kernel_ridge(Xtrn, Xtst, Ytrn):
    """
    Manual implimentation of kernel ridge regression.
    """            
    # Train kernel ridge regression
    print(file_name + ': Training [kernel ridge] regression')
    Ktrn = linear_kernel(Xtrn)
    print('Ktrn shape: '+ str(Ktrn.shape))
    # a = (K+LI)^-1*Y 
    #a = np.dot(np.linalg.inv(Ktrn + my_lam * np.identity(Ktrn.shape[1])), Ytrn)
    a = np.dot(np.linalg.inv(Ktrn + my_lam * np.identity(Ktrn.shape[1])), Ytrn[1:10,::])

    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    print('a shape:' + str(a.shape) + ' Ktst shape:' + str(Ktst.shape))
    #Ytst = np.dot(a.T, Ktst)
    Ytst = a.T * Ktst
    return Ytst

def main():
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y()
    Ytst = kernel_ridge(Xtrn, Xtst, Ytrn)
    output_Ytst(Ytst)
    return 

if __name__ == "__main__":
    main()
