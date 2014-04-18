def readData(f_in_trn, f_in_tst, f_in_sol):
    '''
    This function reads in data as numpy matricies.
    ---
    I: directories of training/testing images
    O: the following data matricies:
    Xtrn, Ytrn, Xtst
    '''
    from readImg import readImg
    import numpy as np
    import os 

    # Init train/test image matrix
    trn_images = []
    tst_images = []

    # Get train/test files from directories
    trn_img_list = [f_in_trn + '/' + f for f in sorted(os.listdir(f_in_trn))]
    tst_img_list = [f_in_tst + '/' + f for f in sorted(os.listdir(f_in_tst))]

    # Map images --> vectors; keep track of progress
    for idx, imgf in enumerate(trn_img_list):
        if (idx % 5000) == 0: print 'on train img: %i' % idx
        trn_images.append(readImg(imgf))
    for idx, imgf in enumerate(tst_img_list):
        if (idx % 5000) == 0: print 'on test img: %i' % idx
        tst_images.append(readImg(imgf))

    # Save images as matrix
    Xtrn = np.vstack(trn_images)
    Xtst = np.vstack(tst_images)
    
    # Read train solutions
    Ytrn = np.loadtxt(open(f_in_sol, 'rb'), delimiter=',', skiprows=1, usecols=(range(1,39))

    return (Xtrn, Ytrn, Xtst)
