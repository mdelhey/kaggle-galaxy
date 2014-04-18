def loadData(f_in, rowskip=0):
    '''
    Load data from a pre-saved flatfile
    '''
    import numpy as np

    X = np.loadtxt(open(f_in, 'rb'), delimiter=',', skiprows=rowskip)
    return X
