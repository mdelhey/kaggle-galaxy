def plotLearningCurve(Xtrn, Ytrn, model, neach = 2500):
    '''
    Plot the learning curve for a given model. This curve is the
    training and test error (via split) of the model as a function
    of the number of observations.
    ---
    I:
    O: Plot of the learning curve.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from splitData import splitData
    from sklearn.metrics import mean_squared_error
    from sklearn import linear_model

    # Split data
    (trnInd, valInd, tstInd) = splitData(Xtrn.shape[0], .70, 0, .30)

    # Subset data
    Xtrn_trn = Xtrn[trnInd]; Ytrn_trn = Ytrn[trnInd]
    Xtrn_tst = Xtrn[tstInd]; Ytrn_tst = Ytrn[tstInd]
    
    # Calculate rmse for neach number of obs
    obs_range = range(10, len(Xtrn_trn), neach)
    trn_err = []; tst_err = []
    for i in obs_range:
        # Fit model. Assume linear for now.
        model.fit(Xtrn_trn[0:i,::], Ytrn_trn[0:i, 1:])
        Ytrn_trn_pred = model.predict(Xtrn_trn[0:i,::])
        Ytrn_tst_pred = model.predict(Xtrn_tst[0:i,::])

        # Calculate train/test RMSE for i
        trn_i_err = np.sqrt(mean_squared_error(Ytrn_trn[0:i, 1:], Ytrn_trn_pred))
        tst_i_err = np.sqrt(mean_squared_error(Ytrn_tst[0:i, 1:], Ytrn_tst_pred))

        # Append to entire train/test vector
        trn_err.append(trn_i_err)
        tst_err.append(tst_i_err)
        
        # Keep us informed on progress, print error
        if ((i-10) % 10000) == 0:
            print 'training example: %i' % i
            print '\t train err: %f \t test err: %f' % (trn_i_err, tst_i_err)
    
    # Plot curve
    plt.plot(obs_range, trn_err)
    plt.plot(obs_range, tst_err)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return
