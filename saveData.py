def saveData(X, f_out, colfmt='%i'):
    '''
    Quick alias for saving data matricies. If X and f_out are tuples,
    this function will save multiple matricies at once.
    '''
    import numpy as np
    
    if isinstance(X, tuple):
        assert(len(X) == len(f_out))
        for idx,Z in enumerate(X):
            np.savetxt(f_out[idx], Z, delimiter=',', fmt=colfmt)

    else:
        np.savetxt(f_out, X, delimiter=',', fmt=colfmt)
