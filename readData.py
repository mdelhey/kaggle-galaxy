def readData(f_in_trn, f_in_tst, f_in_sol):
    '''
    This function reads in data as numpy matricies.
    ---
    I: directories of training/testing images
    O: the following data matricies:
        Xtrn, Ytrn, Xtst
    '''
    import numpy as np
    import os
    from readImg import readImg

    # Init train/test image matrix
    trn_images = []
    tst_images = []

    # Get train/test files from directories
    trn_img_list = [f_in_trn + '/' + f for f in sorted(os.listdir(f_in_trn))]
    tst_img_list = [f_in_tst + '/' + f for f in sorted(os.listdir(f_in_tst))]

    # Map images --> vectors
    for imgf in trn_img_list: trn_images.append(readImg(imgf))
    for imgf in tst_img_list: tst_images.append(readImg(imgf))

    # Save images as matrix
    Xtrn = np.vstack(trn_img_list)
    Xtst = np.vstack(tst_img_list)
    
    # Read train solutions
    Ytrn = np.loadtxt(open(f_in_sol, 'rb'), delimiter = ',', skiprows = 1)

    return (Xtrn, Ytrn, Xtst)
