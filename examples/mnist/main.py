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
    loadResults,
    load,
    train,
    plotResults,
    batchPrediction,
    getClassificationReport,
    getConfusionMatrix
)
from parser import (
    Dataset
)
# external libraries
import torch
import warnings

# ignore warnings, this was added due to PyTorch LazyLayers spamming warnings
warnings.filterwarnings('ignore')
# get a device to run on
device_type = GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE
device = torch.device(device_type)


# Main program.
def main():
    # preprocess
    dataset = Dataset()
    dataset.get()
    dataset.load()
    dataset.normalize()
    dataset.toTensor()
    return

    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()
    trainData, testData = normalizeData(trainData, testData)
    trainLoader = convertDatasetToTensors(device_type, trainData, trainLabels)
    testLoader = convertDatasetToTensors(device_type, testData, testLabels)
    # generate and train model
    loss, accuracy = loadResults()
    model = Model()
    history = train(model, device, device_type, trainLoader, testLoader, loss, accuracy)
    model = load()
    # test model
    plotResults(history)
    yPred, yTrue = batchPrediction(model, testLoader)
    print(getClassificationReport(yPred, yTrue))
    print(getConfusionMatrix(yPred, yTrue))


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
