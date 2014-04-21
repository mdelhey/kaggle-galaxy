def plotLearningCurve(Xtrn, Ytrn, model, param_name, param_range):
    '''
    Plot the bias/variance tradeoff for a given model. This curve is
    the training and test error (via split) of the model as a function
    of model complexity.

    Wrapper for validation_curve in sklearn.
    ---
    I: 
    O: Plot of the bias/var tradeoff.
    '''
    from sklearn.learning_curve import validation_curve
    
    validation_curve(model, Xtrn, Ytrn, param_name, param_range, cv=5,
                     n_jobs=-1, pre_dispatch='all', verbose=1)
    
    
    return
