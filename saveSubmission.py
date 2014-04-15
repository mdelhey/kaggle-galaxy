def forceBounds(a):
    '''
    Elementwise forcing of bounds within [0,1]
    '''
    import numpy as np
    for x in np.nditer(a, op_flags = ['readwrite']):
        if x[...] > 1: x[...] = 1
        if x[...] < 0: x[...] = 0
    return a

def getImgNames(f_in_tst):
    '''
    Grab ID from files in test directory
    '''
    import numpy as np
    import os
    inames = sorted(os.listdir(f_in_tst))
    inames = [int(f.strip('.jpg')) for f in inames]
    return np.asarray(inames)
    
def saveSubmission(Ytst, f_in_tst, f_out, return_Y=False):
    '''
    Prepare Ytst for kaggle. Save as csv.
    '''
    import numpy as np
    from saveData import saveData

    # Get row names (ID Column)
    Ytst_names = getImgNames(f_in_tst)

    # Force [0,1] bounds
    Ytst = forceBounds(Ytst)

    # Concat left-hand-side ID colum
    Ytst = np.c_[Ytst_names, Ytst]

    # Save submission to f_out
    print 'Saving csv to ' + f_out
    colfmt = ['%i'] + ['%.18e'] * (Ytst.shape[1] - 1)
    saveData(Ytst, f_out, colfmt = colfmt)
    #np.savetxt(f_out, Ytst, delimiter = ',', fmt = colfmt)
    
    if (return_Y): return Ytst
    return 
