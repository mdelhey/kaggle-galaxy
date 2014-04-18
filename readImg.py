def readImg(imgf, dim=24):
    '''
    This function loads in an image and computes dim reduction.
    ---
    I: imgf=image file path, dim=downsampled image size, use 64
    O: vector (dim-reduced) representation of the image
    '''
    import numpy as np
    import cv2
    from scipy.cluster.vq import whiten, kmeans

    # Read data file (0 = greyscale, otherwise = rgb)
    img = cv2.imread(imgf, 0)
    
    # Scale data by dividing by 255
    img = img / float(255)

    # Crop images to 200x200
    img = img[112:312, 112:312]

    # Downsample images to 64x64 (dim x dim)
    img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_CUBIC)

    ### Data augmentation
    # Define transformation matrix (2 x 3)
    M = [1 1; 2 2; 3 3]
    cv2.wrapAffine(img, M)
    
    # Flatten data [each col represents the r/g/b color]
    #img = np.reshape(img, (dim*dim,3))
    img = np.reshape(img, (dim*dim))
        
    # Whiten data
    img = whiten(img)

    # k-means to sample each len=16 vector
    #(img, distortion) = kmeans(img, 16)

    # Convert images to vector 
    img = np.reshape(img, np.prod(img.shape))

    return img
    
