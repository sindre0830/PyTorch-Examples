# internal libraries
from dictionary import (
    DATASET_PATH,
    CPU_DEVICE,
    BATCH_SIZE
)
# external libraries
import os
import requests
import gzip
import shutil
import idx2numpy
import numpy as np
import torch
import torch.utils.data
import multiprocessing


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
    device_type: str = None
    xTrain = None
    yTrain = None
    xTest = None
    yTest = None
    train = None
    test = None
    val = None

    def __init__(self, device_type: str = CPU_DEVICE):
        """
        Initializes dataset class.
        """
        self.device_type = device_type

    def toTensor(self):
        self.train = convertDatasetToTensors(self.device_type, self.xTrain, self.yTrain)
        self.test = convertDatasetToTensors(self.device_type, self.xTest, self.yTest)

    def normalize(self):
        self.xTrain = self.xTrain / 255.
        self.xTest = self.xTest / 255.


    def load(self):
        # load numpy data from file
        self.xTrain: np.ndarray = np.load(DATASET_PATH + 'trainData.npy')
        self.yTrain: np.ndarray = np.load(DATASET_PATH + 'trainLabels.npy')
        self.xTest: np.ndarray = np.load(DATASET_PATH + 'testData.npy')
        self.yTest: np.ndarray = np.load(DATASET_PATH + 'testLabels.npy')

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
