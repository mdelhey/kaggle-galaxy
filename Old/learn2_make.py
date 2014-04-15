"""
Takes the ds on disk, preprocess it, saves new ds to disk
"""
import os.path, re, inspect
import pylearn2
from linear import read_X_Y
# For loading the ds
from pylearn2.datasets import npy_npz
# For saving the ds
from pylearn2.utils import serial
# Preprocessing functions
from pylearn2.datasets import preprocessing

# File i/o
f_in_trn = 'Data/train_clr_32.csv'
f_in_tst = 'Data/test_clr_32.csv'
sol_dir = 'Data/train_solutions.csv'
f_out_trn = 'train_clr_32_pp.csv'
f_out_tst = 'test_clr_32_pp.csv'
# Get dim & script name
my_file = re.search(r'_[0-9]+', f_in_trn).group()
my_dim = re.search(r'[0-9]+', my_file).group()
file_name = inspect.getfile(inspect.currentframe())

def main():
    """
    Run experiment
    """
    # Import raw 32x32 color images
    (Xtrn, Xtst, Ytrn, f_out) = read_X_Y(f_in_trn, f_in_tst, sol_dir, my_dim)
    npy_npz.NpyDataset()

    # Setup pp pipeline for multiple opperations
    pipeline = preprocessing.Pipeline()

    # Pull out small patches
    print 'pulling out small patches'
    pipeline.items.append(
        preprocessing.ExtractPatches(patch_shape=(8,8), num_patches=150000)
    )

    # Contrast normalize the patches (w/ regularization params from paper)
    print 'contrast normalizing patches'
    pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))

    # Whiten data using ZCA
    print 'whitening using zca'
    pipeline.items.append(preprocessing.ZCA())

    # Apply pp to dataset. can_fit indicates ZCA allows to fit to dataset
    # I.e. we would use can_fit = False for test data
    print 'applying pp to train (fitting)'
    Xtrn.apply_preprocessor(preprocessor=Pipeline, can_fit=True)
    print 'applying pp to test (not fitting)'
    Xtst.apply_preprocessor(proprocessor=Pipeline, can_fit=False)

    # Set datasets to be saved in Data/ folder
    print 'saving datasets as csv'
    path = pylearn2.__path__[0]
    Xtrn.use_design_loc(os.path.join(path, f_out_trn))
    Xtst.use_design_loc(os.path.join(path, f_out_tst))

    # Also save as pickle file (larger)
    print 'saving datasets as pickle'
    pkl_path_trn = os.path.join(path, re.sub(r'.csv', '.pkl', f_out_trn))
    pkl_path_tst = os.path.join(path, re.sub(r'.csv', '.pkl', f_out_tst))
    serial.save(pkl_path_trn, Xtrn)
    serial.save(pkl_path,tst, Xtst)
    
if __name__ == "__main__":
    main()







