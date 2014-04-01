import numpy as np
import cPickle as pickle
from read import *
from linear import *

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

f_in_trn = 'Data/train_28.csv'
f_in_tst = 'Data/test_28.csv'
sol_dir = 'Data/train_solutions.csv'
f_out_model = 'Models/brain_28_2.pkl'
f_out_subm = 'Submissions/brain_28_2.csv'

my_file = re.search(r'_[0-9]+', f_in_trn).group()
my_dim = re.search(r'[0-9]+', my_file).group()
file_name = inspect.getfile(inspect.currentframe())

# Read data
print file_name + ': reading data'
(Xtrn, Xtst, Ytrn, f_out) = read_X_Y(f_in_trn, f_in_tst, sol_dir, my_dim)

# PARAMETERS
hidden_size = 100
epochs = 600
continue_epochs = 10
val_prop = 0.2

# Prepare dataset
print file_name + ': preparing ds'
Ytrn = Ytrn[:,1:]  # Remove ID col
input_size = Xtrn.shape[1]  # ncols
target_size = Ytrn.shape[1]  # ncols
ds = SupervisedDataSet(input_size, target_size)
ds.setField('input', Xtrn)
ds.setField('target', Ytrn)

# Train a network
print file_name + ': training network'
net = buildNetwork(input_size, hidden_size, target_size, bias = True)
trainer = BackpropTrainer(net, ds)
    
trainer.trainUntilConvergence(verbose = True, validationProportion = val_prop,
                              maxEpochs = epochs, continueEpochs = continue_epochs)

# Save model
print file_name + ': saving model'
pickle.dump(net, open(f_out_model, 'wb'))

# Predict on test data, save to file
print file_name + ': predicting on test data'
Ytst = np.zeros((Xtst.shape[0], target_size))
ds_tst = SupervisedDataSet(input_size, target_size)
ds_tst.setField('input', Xtst)
ds_tst.setField('target', Ytst)

Ytst = net.activateOnDataset(ds_tst)
output_Ytst(Ytst, tst_dir = tst_dir, f_out = f_out_subm)
