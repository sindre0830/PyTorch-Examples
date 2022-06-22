# internal libraries
from functionality import (
    getDataset,
    loadDataset,
    normalizeData,
    convertDatasetToTensors
)
from model import (
    Model,
    train
)
# external libraries
import torch


# Main program.
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()
    trainData, testData = normalizeData(trainData, testData)
    trainDatasetLoader, testDatasetLoader = convertDatasetToTensors(device, trainData, trainLabels, testData, testLabels)
    model = Model()
    train(model, device, trainDatasetLoader)


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
