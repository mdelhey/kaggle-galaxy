def histogramOfGradients(img, bin_n=16):
    '''
    ---
    I:
    O:
    '''
    import numpy as np
    import cv2

    # Calculate Sobel derivatives of each cell in x/y direction
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # Find magnitude/direction of gradient at each pixel
    mag, ang = cv2.cartToPolar(gx, gy)

    # Quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
             for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    
    return hist
