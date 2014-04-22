def readImg(imgf, dim=32, augment=True):
    '''
    This function loads in an image and computes dim reduction.
    ---
    I: imgf=image file path, dim=downsampled image size, use 64
    O: vector (dim-reduced) representation of the image
    ---
    For testing this function, we can use:
    imgf = 'Data/images_train/100008.jpg'    
    '''
    import numpy as np
    import cv2
    from scipy.cluster.vq import whiten, kmeans

    # Read data file (0 = greyscale, otherwise = rgb)
    img = cv2.imread(imgf,0)
    
    # Scale data by dividing by 255
    img = img / float(255)

    # Crop images to 200x200
    img = img[112:312, 112:312]

    # Downsample images to 64x64 (dim x dim)
    img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_CUBIC)

    # Deskew the images
    from deskewImg import deskewImg
    img = deskewImg(img, dim)

    # Data augmentation
    if augment:
        from augmentImg import augmentImg
        img = augmentImg(img, dim)
    
    # Flatten data [each col represents the r/g/b color]
    try:
        img = np.reshape( img, (dim*dim, img.shape[2]) )
    except IndexError:
        img = np.reshape( img, (dim*dim) )
    
    # Whiten data
    img = whiten(img)

    # Convert images to vector 
    img = np.reshape(img, np.prod(img.shape))

    return img
    
