import numpy as np
import cv2, os, inspect, re
import nimfa
from read import *

# Take in feature matrix X for each image.
# Apply nmf to each image

###### Parameters
my_rank = 4
iters = 30
f_in_trn = 'Data/train_50.csv'
f_in_tst = 'Data/test_50.csv'
f_out_trn = 'Data/nmf_train_4.csv'
f_out_tst = 'Data/nmf_test_4.csv'
###### 

def run_nmf():
    file_name = inspect.getfile(inspect.currentframe())
    # Read in pre-processed matricies 
    print(file_name + ': Reading train/test matrix w/ dim = ' + f_in_trn)
    Xtrn = ensure_dim(np.loadtxt(open(f_in_trn, 'rb'), delimiter = ',', skiprows = 0))
    Xtst = ensure_dim(np.loadtxt(open(f_in_tst, 'rb'), delimiter = ',', skiprows = 0))

    # Run nmf
    print(file_name + ': Running non-negative matrix facorization w/ rank = ' + my_rank)
    nmf = nimfa.mf(Xtrn, method = 'nmf', seed = "fixed", max_iter = iters, rank = my_rank,
                   update = 'euclidean', objective = 'fro')
    
    # Output submission
    print(file_name + ': Saving csv to ' + f_out)
    colfmt = ['%i'] + ['%f'] * (Ytst.shape[1] - 1)
    np.savetxt(f_out, Ytst, delimiter = ',', fmt = colfmt)


def main():
    run_nmf()
    
    
if __name__ == "__main__":
    main()
