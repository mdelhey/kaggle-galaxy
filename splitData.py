def splitData(nrows, train = .6, val = .35, test = .05):
    '''
    Randomly partition data into train/val/test.
    ---
    I: Number of rows in data; desired split.
    O: Indicies of split.
    '''
    import numpy as np
    # Sample train: if odd, take ceiling --> more training obs
    rows = np.arange(nrows)
    trainInd = np.random.choice(rows, size=int(np.ceil(nrows*train)), replace=False)
    # Sample val: first remove training rows, take floor --> less val obs
    rows = np.setdiff1d(rows, trainInd)
    valInd = np.random.choice(rows, size=int(np.floor(nrows*val)), replace=False)
    # Sample test: just take rows that aren't in train/val
    rows = np.setdiff1d(rows, valInd)
    testInd = rows
    
    return (trainInd, valInd, testInd)
