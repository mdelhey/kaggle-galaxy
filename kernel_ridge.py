import os, inspect, re
import numpy as np
from read import *
from linear import *

# Take care of some file i/o
my_file = re.search(r'train_[0-9]+', f_in_trn).group()
my_dim = re.search(r'[0-9]+', my_file).group()
file_name = inspect.getfile(inspect.currentframe())
f_out = 'Submissions/ls_nmf_' + str(my_dim) + '.csv'

def linear_kernel(Xtrn, Xtst, c = 0):
    print(file_name + ': \t Using linear kernel')
    #Ktrn = np.dot(Xtrn.T, Xtrn) + c
    Ktrn = np.dot(Xtrn, Xtrn.T) + c
    #Ktst = np.dot(Xtst.T, Xtst) + c
    Ktst = np.dot(Xtst, Xtst.T)
    return (Ktrn, Ktst)

def rbf_kernel(Xtrn, Xtst, sigma = 1):
    print(file_name + ': \t Using radial basis kernel')
    return (Ktrn, Ktst)

def kernel_ridge(Xtrn, Xtst, Ytrn):
    """
    Manual implimentation of kernel ridge regression.
    """            
    # Train kernel ridge regression
    print(file_name + ': Training [kernel ridge] regression')
    (Ktrn, Ktst) = linear_kernel(Xtrn, Xtst)
    print('Ktrn shape: '+ str(Ktrn.shape))
    # a = (K+LI)^-1*Y 
    #a = np.dot(np.linalg.inv(Ktrn + my_lam * np.identity(Ktrn.shape[1])), Ytrn)
    a = dot(np.linalg.inv(Ktrn + my_lam * np.identity(Ktrn.shape[1])), Ytrn[::,0])

    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    print('a shape:' + str(a.shape) + ' Ktst shape:' + str(Ktst.shape))
    #Ytst = np.dot(a.T, Ktst)
    Ytst = a.T * Ktst
    return Ytst

def main():
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y(f_in_trn, f_in_tst)
    Ytst = kernel_ridge(Xtrn, Xtst, Ytrn)
    output_Ytst(Ytst)
    return 

if __name__ == "__main__":
    main()
