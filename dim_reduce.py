import numpy as np
import nimfa
from linear import *

# Take in feature matrix X for each image.
# Apply nmf to each image

###### Parameters
my_rank = 4
iters = 30
f_in_trn = 'Data/train_28.csv'
f_in_tst = 'Data/test_28.csv'
f_out_trn = 'Data/nmf_train_4.csv'
f_out_tst = 'Data/nmf_test_4.csv'
######
file_name = inspect.getfile(inspect.currentframe())

def output_X(Xtrn, Xtst):
    """
    Output factorized (low-dim) X matricies
    """
    print(file_name + ': Saving csv to ' + f_out_trn + 'and ' + f_out_tst)
    np.savetxt(f_out_trn, Xtrn, delimiter = ',', fmt = '%i')
    np.savetxt(f_out_tst, Xtst, delimiter = ',', fmt = '%i')
    return 

def pca_white():
    return

def nmf(Xtrn, Xtst):
    # Init matricies
    Xtrn_n = np.shape(Xtrn)[0]
    Xtst_n = np.shape(Xtst)[0]
    Xtrn_nmf = np.zeros((Xtrn_n, my_rank))
    Xtst_nmf = np.zeros((Xtst_n, my_rank))
    print(file_name + ': Running non-negative matrix facorization w/ rank = ' + str(my_rank))
    #Xtrn_fctr = nimfa.mf(Xtrn, method = 'nmf', seed = "fixed", max_iter = iters,
    #                     rank = my_rank, update = 'euclidean', objective = 'fro')
    print(file_name + ': \t on traning...')
    for i in xrange(Xtrn_n):
        Xtrn_fctr = nimfa.mf(Xtrn[i,:], method = 'lsnmf', max_iter = iters, rank = my_rank)
        Xtrn_res = nimfa.mf_run(Xtrn_fctr)
        Xtrn_nmf[i,:] = Xtrn_res.basis()
        if (i%10000 == 0): print(file_name + ': \t iter ' + str(i))
    print(file_name + ' \t on testing...')
    for i in xrange(Xtst_n):
        Xtst_fctr = nimfa.mf(Xtst[i,:], method = 'lsnmf', max_iter = iters, rank = my_rank)
        Xtst_res = nimfa.mf_run(Xtrn_fctr)
        Xtst_nmf[i,:] = Xtst_res.basis()
        if (i%10000 == 0): print(file_name + ': \t iter ' + str(i))
    
    """
    Xtrn_sm = Xtrn_res.summary()
    Xtst_sm = Xtst_res.summary()
    print(file_name + ': \t\t RSS \t Explained Var \t Iters')
    print(file_name + ': Xtrn: \t' + str(Xtrn_sm['rss']) + '\t' +
          str(Xtrn_sm['evar']) + '\t' + str(Xtrn_sm['n_iter']))
    print(file_name + ': Xtst: ' + str(Xtst_sm['rss']) + '\t' +
          str(Xtst_sm['evar']) + '\t' + str(Xtst_sm['n_iter']))
    """
    
    return (Xtrn_nmf, Xtst_nmf)

def run_nmf():
    # Read in pre-processed matricies
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y()
    # Compute factorization
    (Xtrn_nmf, Xtst_nmf) = nmf(Xtrn, Xtst)     
    # Output csv
    output_X(Xtrn_nmf, Xtst_nmf)
    return 
    
def main():
    run_nmf()
    return 
    
    
if __name__ == "__main__":
    main()
