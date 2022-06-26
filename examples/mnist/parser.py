# internal libraries
from dictionary import (
    DATASET_PATH,
    TEST_SIZE
)
# external libraries
import os
import requests
import gzip
import shutil
import idx2numpy
import numpy as np


class Dataset():
    """
    Class for parsing datasets for PyTorch machine learning purposes.
    """
    urls = {
        'trainData': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'trainLabels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'testData': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'testLabels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    xTrain = None
    yTrain = None
    xTest = None
    yTest = None
    trainDataset = None
    testDataset = None
    valDataset = None

    def __init__(self):
        """
        Initializes dataset class.
        """
        pass

    def normalize(self):
        self.xTrain = self.xTrain / 255.
        self.xTest = self.xTest / 255.


    def load(self):
        # load numpy data from file
        self.xTrain = np.load(DATASET_PATH + 'trainData.npy')
        self.yTrain = np.load(DATASET_PATH + 'trainLabels.npy')
        self.xTest = np.load(DATASET_PATH + 'testData.npy')
        self.yTest = np.load(DATASET_PATH + 'testLabels.npy')

    def get(self):
        # create path if it doesn't exist
        os.makedirs(DATASET_PATH, exist_ok=True)
        # iterate through each url
        for filename, url in self.urls.items():
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
