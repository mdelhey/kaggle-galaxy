import numpy as np
from linear import read_X_Y, output_Ytst
from read import ensure_dim, force_bounds, get_image_names

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
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y()
    Ytst = kernel_ridge(Xtrn, Xtst, Ytrn)
    output_Ytst(Ytst)
    return 

if __name__ == "__main__":
    main()
