f_in_trn = 'Data/images_train'
f_in_tst = 'Data/images_test'
f_in_sol = 'Data/train_solutions.csv'

f_out_trn = 'Data/train_24.csv'
f_out_tst = 'Data/test_24.csv'
f_out_subm = 'Submissions/new_pp_test.csv'

from readData import readData
(Xtrn, Ytrn, Xtst) = readData(f_in_trn, f_in_tst, f_in_sol)

from saveData import saveData
saveData((Xtrn, Xtst), (f_out_trn, f_out_tst), colfmt='%.18e')
  
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(Xtrn, Ytrn)
Ytst = model.predict(Xtst)

from saveSubmission import *
saveSubmission(Ytst, f_in_tst, f_out_subm)
