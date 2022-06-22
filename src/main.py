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
import warnings

# ignore warnings, this was added due to PyTorch LazyLayers causing warning
warnings.filterwarnings('ignore')
# get a device to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Main program.
def main():
    # preprocess
    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()
    trainData, testData = normalizeData(trainData, testData)
    trainDatasetLoader = convertDatasetToTensors(trainData, trainLabels)
    # create model
    model = Model()
    train(model, device, trainDatasetLoader)


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
