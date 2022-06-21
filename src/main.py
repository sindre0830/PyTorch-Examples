# internal libraries
from functionality import (
    getDataset,
    loadDataset,
    convertDatasetToTensors
)


# Main program.
def main():
    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()
    trainDatasetLoader, testDatasetLoader = convertDatasetToTensors(trainData, trainLabels, testData, testLabels)


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
