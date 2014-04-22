def deskewImg(img, dim):
    '''
    Takes an image and properly aligns it.
    ---
    I: Image matrix from cv2; dim of image.
    O: Image matrix, deskewed. 
    '''
    import numpy as np
    import cv2
    
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

    # Calculate moments
    m = cv2.moments(img)

    # If no moments, return plain image
    if abs(m['mu02']) < 1e-2:
        return img.copy()

    # Calculate skew
    skew = m['mu11'] / m['mu02']

    # Create affine transform from skew
    M = np.float32([[1, skew, -0.5*dim*skew], [0, 1, 0]])

    # Appliy skew-transform to matrix
    img = cv2.warpAffine(img,M,(dim, dim),flags=affine_flags)
    
    return img
