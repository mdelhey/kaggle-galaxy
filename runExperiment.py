f_in_trn = 'Data/images_train'
f_in_tst = 'Data/images_test'
f_in_sol = 'Data/train_solutions.csv'

f_in_flat_trn = 'Data/train_64_hog.csv'
f_in_flat_tst = 'Data/test_64_hog.csv'

f_out_trn = 'Data/train_64_hog.csv'
f_out_tst = 'Data/test_64_hog.csv'
f_out_subm = 'Submissions/ls_64_hog.csv'

# Process images (save to disk)
from readData import readData
(Xtrn, Ytrn, Xtst) = readData(f_in_trn, f_in_tst, f_in_sol, augmenting=False)
from saveData import saveData
saveData((Xtrn, Xtst), (f_out_trn, f_out_tst), colfmt='%.18e')

# Load processed images from flat file, on disk
'''
from loadData import loadData
Xtrn = loadData(f_in_flat_trn, rowskip=0)
Xtst = loadData(f_in_flat_tst, rowskip=0)
tst = loadData(f_in_flat_tst, rowskip=0)
Ytrn = loadData(f_in_sol, rowskip=1)
'''

# Fit OLS
print 'fitting ols'
from scipy.cluster.vq import whiten
Xtrn = whiten(Xtrn)
Xtst = whiten(Xtst)
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(Xtrn, Ytrn[::, 1:])
Ytst = model.predict(Xtst)

# Fit RF
'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(Xtrn, Ytrn[::, 1:])
Ytst = model.predict(Xtst)
'''

# Fit Extra Trees (More random)
'''
seed = 0
from sklearn.ensemble import ExtraTreesRegressor
'''
# n_estimators=10 : 0.14191 [sqrt]
# n_estimators=50 : 0.13590 [sqrt]
# n_estimators=50 : 0.13081 [None]
# n_estimators=500 : 0.12935 [None]
# n_estimators=10 : 0.14191 [sqrt] [32]         
# n_estimators=10 : 0.14185 [sqrt] [32c]        
# n_estimators=10 : 0.13998 [sqrt] [64]         
# n_estimators=50 : 0.13590 [sqrt]              
# n_estimators=50 : 0.13081 [None]              
# n_estimators=500 : 0.12954 [None]             
# n_estimators=500 : 0.12935 [None]
'''
model = ExtraTreesRegressor(n_estimators=500, max_features=None,
                            random_state=seed, verbose=True,
                            oob_score=True, bootstrap=True,
                            n_jobs=8)
model.fit(Xtrn, Ytrn[::, 1:])
Ytst = model.predict(Xtst)
'''

# Plot learning curve
'''
from plotLearningCurve import plotLearningCurve
plotLearningCurve(Xtrn, Ytrn, model)
'''

# Fit Ridge
'''
model = linear_model.RidgeCV(alphas = [0, 25, 50, 100])
model.fit(Xtrn, Ytrn[::, 1:])
my_alpha = model.alpha_
model = linear_model.Ridge(alpha = my_alpha)
model.fit(Xtrn, Ytrn[::, 1:])
Ytst = model.predict(Xtst)
'''

# Save submission to disk
from saveSubmission import *
saveSubmission(Ytst, f_in_tst, f_out_subm)
