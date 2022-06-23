# internal libraries
from dictionary import (
    CPU_DEVICE,
    DATASET_PATH,
    BATCH_SIZE
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
import multiprocessing
import tqdm


# Set prefix in progressbar and update output.
def setProgressbarPrefix(progressbar: tqdm.tqdm, trainLoss: float = 0., trainAccuracy: float = 0., valLoss: float = 0., valAccuracy: float = 0.):
    trainLossStr = f'Train loss: {trainLoss:.4f}, '
    trainAccuracyStr = f'Train acc: {trainAccuracy:>7.4f}, '
    valLossStr = f'Val loss: {valLoss:.4f}, '
    valAccuracyStr = f'Val acc: {valAccuracy:>7.4f}'
    progressbar.set_postfix_str(trainLossStr + trainAccuracyStr + valLossStr + valAccuracyStr)


# Generates progressbar for iterable used in model training.
def getProgressbar(iter: torch.utils.data.DataLoader, epoch, epochs):
    width = len(str(epochs))
    progressbar = tqdm.tqdm(
        iterable=iter,
        desc=f'Epoch {(epoch + 1):>{width}}/{epochs}',
        ascii='░▒',
        unit=' steps',
        colour='blue'
    )
    setProgressbarPrefix(progressbar)
    return progressbar


# Converts dataset to a PyTorch tensor dataset.
def convertDatasetToTensors(device_type: str, data: np.ndarray, labels: np.ndarray):
    # reshape data by adding channels
    data = np.expand_dims(data, axis=1).astype('float32')
    # convert to tensors
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    # convert to dataset
    dataset = torch.utils.data.TensorDataset(data, labels)
    # convert to data loader
    pin_memory = False
    workers = 0
    # branch if device is set to CPU and set parameters accordingly
    if device_type is CPU_DEVICE:
        pin_memory = True
        workers = multiprocessing.cpu_count()
    datasetLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=pin_memory, num_workers=workers)
    return datasetLoader


# Normalizes the data from 0-255 to 0-1.
def normalizeData(trainData: np.ndarray, testData: np.ndarray):
    trainData = trainData / 255.
    testData = testData / 255.
    return trainData, testData


# Loads dataset from dataset directory
def loadDataset():
    trainData = np.load(DATASET_PATH + 'trainData.npy')
    trainLabels = np.load(DATASET_PATH + 'trainLabels.npy')
    testData = np.load(DATASET_PATH + 'testData.npy')
    testLabels = np.load(DATASET_PATH + 'testLabels.npy')
    return trainData, trainLabels, testData, testLabels


# Downloads and extracts the MNIST dataset.
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
