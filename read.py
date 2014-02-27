import numpy as np
import cv2, os, inspect

# X = individual images, n = 61,578, p = my_dim x my_dim
# Y = response for each, length(y) = 37

###### Parameters
## Set dimention of image matrix (my_dim x my_dim)
my_dim = 424
f_out_trn = 'Data/train'
f_out_tst = 'Data/test'
trn_dir = 'Data/images_train'
tst_dir = 'Data/images_test'
sol_dir = 'Data/train_solutions.csv'
######
file_name = inspect.getfile(inspect.currentframe())


def read_images_grey(dir, dim = 128, n = 0):
    # Read in images as matrix
    # Shrink 424x424 --> dim x dim
    images = []
    image_files = sorted(os.listdir(dir))
    image_files = [dir + '/' + f for f in image_files]
    if n != 0:
        image_files = image_files[0:n]
    for imgf in image_files:
        img = cv2.imread(imgf, 0)
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_CUBIC)
        length = np.prod(img.shape)
        img = np.reshape(img, length)
        images.append(img)
    images = np.vstack(images) # Save images as matrix
    return images

def read_images_vector(dir, crop = 140, pixel_size = 5):
    #import PIL from Image
    import glob
    crop_dimensions = (crop, crop, 284, 284)
    glob_files = dir + '/*.jpg'
    #open a file to store our vectors
    with open(vector_file, "wb") as outfile:
        for enum, infile in enumerate( glob.glob(glob_files) ):
            #open image and get the filename and extension
            image = Image.open(infile)
            filen, ext = os.path.splitext(infile)
            file_id = filen[-6:]
            # take out interesting part in the center-center, leaves 140x140 image
            image = image.crop(crop_dimensions)
            # convert to grayscale 0-256
            image = image.convert('L')
            # pixelate the image with pixel_size
            image = image.resize((image.size[0]/pixel_size, image.size[1]/pixel_size), Image.NEAREST)
            image = image.resize((image.size[0]*pixel_size, image.size[1]*pixel_size), Image.NEAREST)
            # load resulting pixel data
            pixel = image.load()
            # convert every pixelated block to a 0-1 float
            u = 0
            vectors = []
            for i in xrange(0,image.size[0],pixel_size):
                for y in xrange(0,image.size[1],pixel_size):
                    vectors.append( (u, round(pixel[y, i]/float(255),3)) )
                    u += 1
            #write vectors to file (or format as libSVM or whatever)
            outfile.write(str(file_id) + " " + str(vectors) + "\n")
            #status report
            if enum % 100 == 0:
                print (file_name + ': \t on iter ' + str(enum))
    return vectors

def get_image_names(dir, n = 0):
    inames = sorted(os.listdir(dir))
    inames = [int(f.strip('.jpg')) for f in inames]
    if n != 0:
        inames = inames[0:n]
    return np.asarray(inames)

def force_bounds(a):
    for x in np.nditer(a, op_flags = ['readwrite']):
        if x[...] > 1:
            x[...] = 1
        if x[...] < 0:
            x[...] = 0
    return a

def ensure_dim(a):
    try:
        x = a.shape[1]
    except IndexError:
        print('\t Dim is 1! Ensuring a multi-dim array')
        a = np.reshape(a, (len(a), 1))
    return a

def run_grey():
    print(file_name + ': using dim = ' + str(my_dim))
    print(file_name + ': Reading training images') 
    Xtrn = read_images_grey(trn_dir, dim = my_dim)
    print(file_name + ': Reading test images')
    Xtst = read_images_grey(tst_dir, dim = my_dim)
    print(file_name + ': Saving .csv files')
    f_trn = f_out_trn + '_' + str(my_dim) + '.csv'
    f_tst = f_out_tst + '_' + str(my_dim) + '.csv'
    np.savetxt(f_trn, Xtrn, delimiter = ',', fmt = '%i')
    np.savetxt(f_tst, Xtst, delimiter = ',', fmt = '%i')
    
def run_vec():
    print(file_name + ': Converting images into p = 783 vector')
    print(file_name + ': Training data...')
    Xtrn = read_images_vector(trn_dir, crop = 140, pixel_size = 5)
    print(file_name + ': Test data...')
    Xtst = read_images_vector(tst_dir, crop = 140, pixel_size = 5)
    print(file_name + ': Saving .csv files')
    f_trn = f_out_trn + '_vec.csv'
    f_tst = f_out_tst + '_vec.csv'
    np.savetxt(f_trn, Xtrn, delimiter = ',', fmt = '%i')
    np.savetxt(f_tst, Xtst, delimiter = ',', fmt = '%i')

def main():
    run_grey()
    #run_vec()
    
if __name__ == "__main__":
    main()
