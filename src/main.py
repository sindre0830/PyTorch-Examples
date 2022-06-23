# internal libraries
from dictionary import (
    GPU_DEVICE,
    CPU_DEVICE
)
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
device_type = GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE
device = torch.device(device_type)


# Main program.
def main():
    # preprocess
    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()
    trainData, testData = normalizeData(trainData, testData)
    trainLoader = convertDatasetToTensors(device_type, trainData, trainLabels)
    testLoader = convertDatasetToTensors(device_type, testData, testLabels)
    # create model
    model = Model()
    train(model, device, device_type, trainLoader, testLoader)


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
