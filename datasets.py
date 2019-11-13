import os
import gzip
import numpy as np
from urllib.request import urlretrieve
from urllib.parse import urljoin
import random

random.seed(1)
np.random.seed(1)

# Dataset download path
mnist_url =  'http://yann.lecun.com/exdb/mnist/'

# Get the current working directory
cwd = os.getcwd()

# Create a path for the data
if os.path.isdir(os.path.join(cwd, 'data')):
    pass
else:
    os.mkdir('data')

def mnist(noTrSamples=1000, noValSamples=1000, noTsSamples=100, digits=[3, 8]):
    # Reading images and labels stored in binary format
    data_dir = os.getcwd() + '/data/'
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    # Normalize the data
    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    valX = np.zeros((noValSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)
    valY = np.zeros(noValSamples)

    # Extracts equal number of examples per class
    # If number of samples asked and number of classes are not balanced, exception is raised
    if noTrSamples % len(digits) != 0:
        raise ValueError("Unequal number of samples per class will be returned, adjust noTrSamples and digits accordingly!")
    else:
        noTrPerClass = noTrSamples // len(digits)

    if noTsSamples % len(digits) != 0:
        raise ValueError("Unequal number of samples per class will be returned, adjust noTsSamples and digits accordingly!")
    else:
        noTsPerClass = noTsSamples // len(digits)
        
    if noValSamples % len(digits) != 0:
        raise ValueError("Unequal number of samples per class will be returned, adjust noTsSamples and digits accordingly!")
    else:
        noValPerClass = noValSamples // len(digits)

    count = 0
    for ll in range(len(digits)):
        # Train data
        idl = np.where(trLabels == digits[ll])[0]
        np.random.shuffle(idl)
        idl_ = idl[: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl_, :]
        trY[idx] = trLabels[idl_]
        
        valIdl = idl[noTrPerClass:noTrPerClass+noValPerClass]
        idx = list(range(count*noValPerClass, (count+1)*noValPerClass))
        valX[idx, :] = trData[valIdl, :]
        valY[idx] = trLabels[valIdl]
        
        # Test data
        idl = np.where(tsLabels == digits[ll])[0]
        np.random.shuffle(idl)
        idl = idl[: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
        
    train_idx = np.random.permutation(trX.shape[0])
    trX = trX[train_idx,:]
    trY = trY[train_idx]

    val_idx = np.random.permutation(valX.shape[0])
    valX = valX[val_idx,:]
    valY = valY[val_idx]
    
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    valX = valX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    valY = valY.reshape(1,-1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY, valX, valY

#variable to the data folder path
datapath = cwd + '/data/'

# Function to download data from official website and unzip
# Checks if the data file already exists and downloads the data and unzips it only if it doesn't already exist
def download_parse(fgz):
    if os.path.exists(os.path.join(datapath, fgz)):
        pass
    else:
        url = urljoin(mnist_url, fgz)
        filename = os.path.join(datapath, fgz)
        urlretrieve(url, filename)
        os.system('gunzip ' + filename)

# Paths of train and test images and labels
download_parse('train-images-idx3-ubyte.gz')
download_parse('t10k-images-idx3-ubyte.gz')
download_parse('train-labels-idx1-ubyte.gz')
download_parse('t10k-labels-idx1-ubyte.gz')

# Example command for how the function works
# trainX, trainY, testX, testY = mnist(noTrSamples=500, noTsSamples=200, digits=[3,8,9,2])
