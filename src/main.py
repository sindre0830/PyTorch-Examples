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


# Main program.
def main():
    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()
    trainData, testData = normalizeData(trainData, testData)
    trainDatasetLoader, testDatasetLoader = convertDatasetToTensors(trainData, trainLabels, testData, testLabels, batch_size=32)
    model = Model()
    print(model)
    train(model, trainDatasetLoader, epochs=30)


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
