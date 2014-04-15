def readImg(imgf):
    '''
    This function loads in an image and computes dim reduction.
    ---
    I: image file path
    O: vector (dim-reduced) representation of the image
    '''
    from scipy.cluster.vq import whiten, kmeans
    import cv2

    # Read data file
    img = cv2.imread(imgf)
    
    # Scale data by dividing by 255
    img = img / float(255)

    # Crop images to 200x200
    img = img[112:312, 112:312]

    # Downsample images to 64x64
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    ### Data augmentation!

    # Flatten data [each col represents the r/g/b color]
    img = np.reshape(img, (64*64,3))
        
    # Whiten data
    img = whiten(img)

    # k-means to sample each len=16 vector
    (img, distortion) = kmeans(img, 16)

    # Convert images to vector 
    img = np.reshape(img, np.prod(img.shape))

    return img
    
