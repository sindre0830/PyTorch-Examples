# internal libraries
from functionality import (
    getDataset,
    loadDataset
)


# Main program.
def main():
    getDataset()
    trainData, trainLabels, testData, testLabels = loadDataset()


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
