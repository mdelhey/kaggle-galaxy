def augmentImg(img, dim):
    '''
    This function randomly applies augmentation to the data.
    '''
    import numpy as np
    import cv2
    from sklearn.decomposition import PCA
    
    # Define center of rotation
    center = (dim/2, dim/2)

    # Add translation? Random shift between -4 and 4 pixels,
    # relative to the original image size of 424x424 in x and y
    # direction, uniform

    # Random angle in [0, 360] (uniform)
    angle = np.random.uniform(low=0, high=360)

    # Random scale between 1/1.3 and 1.3 (log-uniform)
    scale = np.random.normal(1, 0.1)

    # Flip image?
    if np.random.binomial(1, 0.5) == 1:
        angle = angle + 180
    
    # Construct rotation matrixs
    rotMat = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Do affine transformation
    imgDst = cv2.warpAffine(img, rotMat, img.shape)

    # Color perturbation
    color_scale = np.random.normal(0, 0.4)
    pca = PCA(n_components = 1, copy=True, whiten=False)
    pca.fit(imgDst)

    return imgDst
