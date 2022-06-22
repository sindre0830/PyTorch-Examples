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
import torch
import torch.utils.data


# Converts dataset to a PyTorch tensor dataset.
def convertDatasetToTensors(trainData: np.ndarray, trainLabels: np.ndarray, testData: np.ndarray, testLabels: np.ndarray, batch_size: int):
    xTrainTensor = torch.tensor(trainData)
    yTrainTensor = torch.tensor(trainLabels)
    xTestTensor = torch.tensor(testData)
    yTestTensor = torch.tensor(testLabels)
    trainDataset = torch.utils.data.TensorDataset(xTrainTensor, yTrainTensor)
    testDataset = torch.utils.data.TensorDataset(xTestTensor, yTestTensor)
    trainDatasetLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=1)
    testDatasetLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return trainDatasetLoader, testDatasetLoader


# Normalizes the data from 0-255 to 0-1.
def normalizeData(trainData: np.ndarray, testData: np.ndarray):
    trainData = trainData / 255.
    testData = testData / 255.
    trainData = trainData.reshape(trainData.shape[0], trainData.shape[1], trainData.shape[2], 1).astype('float32')
    testData = testData.reshape(testData.shape[0], testData.shape[1], testData.shape[2], 1).astype('float32')
    return trainData, testData


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
