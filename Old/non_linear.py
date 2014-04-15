from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

def knn(Xtrn, Xtst, Ytrn):
    # Train knn regression
    print(file_name + ': Training [knn] regression')
    model = sk.neighbors.KNeighborsRegressor(n_neighbors = 5, weights = 'distance')
    model.fit(Xtrn, Ytrn[::, 1:])
    
    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst

def svm(Xtrn, Xtst, Ytrn):
    # Train svm regression
    print(file_name + ': Training [svm] regression')
    model = sk.svm.SVR(kernel = 'linear')
    model.fit(Xtrn, Ytrn[::, 1:])

    # Predict on test matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst

def rf(Xtrn, Xtst, Ytrn, method = "rf"):
    if method == "rf":
        print(file_name + ': Traning [rf] regression')
        model = RandomForestRegressor()
    if method == "extra":
        print(file_name + ': Training [Extra Trees] regression')
        model = ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0)
    model.fit(Xtrn, Ytrn[::, 1:])
    # Predict on text matrix
    print(file_name + ': Predicting on test matrix')
    Ytst = model.predict(Xtst)
    return Ytst
