f_in_trn = 'Data/images_train'
f_in_tst = 'Data/images_test'
f_in_sol = 'Data/train_solutions.csv'

from readData import readData
(Xtrn, Ytrn, Xtst) = readData(f_in_trn, f_in_tst, f_in_sol)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(Xtrn, Ytrn)
model.predict(Xtst)

