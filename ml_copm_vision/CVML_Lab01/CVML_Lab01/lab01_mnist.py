
import load_mnist
from matplotlib import pyplot as plt
import numpy as np

def main():

    # 1. Use 'load_mnist.loadMnist' function to read 60000 training images and class labels
    #    of MNIST digits (files 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte').
    
    # 2. Preview one image from each class.
    
    # 3. Transform the image data, such that each image forms a row vector,
    #    - NOTE: Math in lectures assumes the column-format, exercises will assume the row-format
    #      (Row-format is used by most Python libraries). This means that we will have to "transpose"
    #      formulae before using them.
    
    # 4. Save the image matrix and the labels in a numpy .npy or .npz files.
    #    'np.save'/'np.savez'/'np.savez_compressed', loading is done using 'np.load'
    #    - 'np.savez_compressed' is the most efficient.
    
    # 5. Do the same for 10000 test digits (files 't10k-images.idx3-ubyte' and
    #    't10k-labels.idx1-ubyte')
    #    - Both files (training and testing set) will be used during the semester.
    
    # 6. Now, try to load the created files, display some of the images and print their respective
    #    labels.
    
    pass

if __name__ == "__main__":
    main()
