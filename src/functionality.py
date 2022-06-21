# internal libraries
from dictionary import (
    DATASET_PATH
)
# external libraries
import requests
import os
import gzip
import shutil
import idx2numpy
import numpy as np


# Loads dataset from dataset directory
def loadDataset():
    trainData = np.load(DATASET_PATH + 'trainData.npy')
    trainLabels = np.load(DATASET_PATH + 'trainLabels.npy')
    testData = np.load(DATASET_PATH + 'testData.npy')
    testLabels = np.load(DATASET_PATH + 'testLabels.npy')
    return trainData, trainLabels, testData, testLabels


# Downloads and extracts the MNIST dataset from the internet.
def getDataset():
    # create direcotries if they don't exists
    os.makedirs(DATASET_PATH, exist_ok=True)
    urls = {
        'trainData': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'trainLabels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'testData': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'testLabels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    # iterate through each url
    for filename, url in urls.items():
        # branch if file already exists
        if not os.path.exists(DATASET_PATH + filename + '.npy'):
            # request compressed file
            req = requests.get(url, allow_redirects=True)
            open(DATASET_PATH + filename + '.gz', 'wb').write(req.content)
            # uncompress file
            with gzip.open(DATASET_PATH + filename + '.gz', 'rb') as source, open(DATASET_PATH + filename, 'wb') as dest:
                shutil.copyfileobj(source, dest)
            os.remove(DATASET_PATH + filename + '.gz')
            # convert to npy
            arr: np.ndarray = idx2numpy.convert_from_file(DATASET_PATH + filename)
            np.save(DATASET_PATH + filename + '.npy', arr)
            os.remove(DATASET_PATH + filename)
