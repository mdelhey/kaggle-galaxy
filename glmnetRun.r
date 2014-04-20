library(glmnet)

f_in_flat_trn = 'Data/train_24.csv'
f_in_flat_tst = 'Data/test_24.csv'
f_in_sol = 'Data/train_solutions.csv'
f_out_subm = 'Submissions/glmnet.csv'

# Read data
Xtrn <- read.table(f_in_flat_trn, header=FALSE, sep=",",
                   colClasses = c(numeric), nrows =  61578)
Xtst <- read.csv(f_in_flat_tst, header=FALSE)
Ytrn <- read.csv(f_in_sol, header=TRUE)

mfit <- glmnet(Xtrn, Ytrn, family = "mgaussian", standardize = FALSE)
plot(mfit, xvar = "lambda", label = TRUE, type.coef = "2norm")
