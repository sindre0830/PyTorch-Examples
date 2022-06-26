# internal libraries
from dictionary import (
    CPU_DEVICE,
    MODEL_PATH,
    PLOT_PATH,
    EPOCHS,
    BATCH_SIZE,
    CHANNELS,
    LABELS_TOTAL,
    GPU_DEVICE,
    LABELS_NAME
)
from functionality import (
    getProgressbar,
    setProgressbarPrefix
)
# external libraries
import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import pandas as pd


# Save model.
def save(model: torch.nn.Module):
    torch.save(model.state_dict(), MODEL_PATH)


# Generate confusion matrix.
def getConfusionMatrix(yPred: np.ndarray, yTrue: np.ndarray):
    # flatten array to get index of highest value
    yPred = np.argmax(yPred, axis=1)
    # calculate and return confusion matrix
    return pd.DataFrame(sklearn.metrics.confusion_matrix(yTrue, yPred), index=LABELS_NAME, columns=LABELS_NAME)


# Generate classification report.
def getClassificationReport(yPred: np.ndarray, yTrue: np.ndarray):
    # flatten array to get index of highest value
    yPred = np.argmax(yPred, axis=1)
    # calculate and return classification report
    return sklearn.metrics.classification_report(yTrue, yPred, target_names=LABELS_NAME)


# Prediction on dataset.
# Used for classification report and confusion matrix.
def batchPrediction(model: torch.nn.Module, dataset: torch.utils.data.DataLoader):
    yPred = np.empty(shape=(0, LABELS_TOTAL))
    yTrue = np.empty(shape=0)
    model.eval()
    for (data, labels) in dataset:
        output = model(data)
        yPred = np.append(yPred, output.detach().to(CPU_DEVICE).numpy(), axis=0)
        yTrue = np.append(yTrue, labels.detach().to(CPU_DEVICE).numpy(), axis=0)
    yPred = yPred.astype(np.uint)
    yTrue = yTrue.astype(np.uint)
    return yPred, yTrue


# Plot model results stored in history dict.
def plotResults(history: dict[str, list]):
    # create direcotries if they don't exists
    os.makedirs(PLOT_PATH, exist_ok=True)
    # get values from results
    epochs = range(1, (len(history['train_loss']) + 1))
    # plot training and validation loss
    plt.clf()
    plt.plot(epochs, history['train_loss'], label='Training loss', c='lightgreen')
    plt.plot(epochs, history['validation_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(PLOT_PATH + 'loss.png')
    plt.show()
    # plot validation accuracy
    plt.clf()
    plt.plot(epochs, history['validation_accuracy'], label='Validation accuracy', c='red')
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(PLOT_PATH + 'accuracy.png')
    plt.show()


# Train model defined in the Model class.
def train(
    model: torch.nn.Module,
    device: torch.cuda.device,
    device_type: str,
    trainLoader: torch.utils.data.DataLoader,
    validationLoader: torch.utils.data.DataLoader
):
    # branch if the device is set to GPU and send the model to the device
    if device_type is GPU_DEVICE:
        model.cuda(device)
    # set optimizer and criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    TRAIN_SIZE = len(trainLoader.dataset)
    # init history lists
    arrTrainLoss = []
    arrTrainAccuracy = []
    arrValidationLoss = []
    arrValidationAccuracy = []
    # loop through each epoch
    for epoch in range(EPOCHS):
        correct = 0.
        totalLoss = 0.
        runningLoss = 0.
        # define the progressbar
        progressbar = getProgressbar(trainLoader, epoch, EPOCHS)
        # set model to training mode
        model.train()
        # loop through the dataset
        for i, (data, labels) in enumerate(progressbar):
            # send dataset to device
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # clear gradients
            optimizer.zero_grad()
            # get results
            output = model(data)
            # compute gradients through backpropagation
            loss = criterion(output, labels)
            loss.backward()
            # apply gradients
            optimizer.step()
            # calculate running loss
            runningLoss += loss.item()
            totalLoss += loss.item()
            # calculate accuracy
            output = torch.argmax(output, dim=1)
            correct += (output == labels).float().sum()
            # branch if iteration is on the last step and validate model then update the information with the final values
            if i >= (TRAIN_SIZE / BATCH_SIZE) - 1:
                validationLoss, validationAccuracy = validateTraining(model, device, criterion, validationLoader)
                trainLoss = totalLoss / TRAIN_SIZE
                trainAccuracy = correct / TRAIN_SIZE
                setProgressbarPrefix(progressbar, trainLoss, trainAccuracy, validationLoss, validationAccuracy)
                # store epoch results
                arrTrainLoss.append(trainLoss)
                arrTrainAccuracy.append(trainAccuracy)
                arrValidationLoss.append(validationLoss)
                arrValidationAccuracy.append(validationAccuracy)
            # branch if batch size is reached and update information with current values
            elif i % BATCH_SIZE == (BATCH_SIZE - 1):
                trainLoss = runningLoss / (TRAIN_SIZE / BATCH_SIZE)
                trainAccuracy = correct / TRAIN_SIZE
                setProgressbarPrefix(progressbar, trainLoss, trainAccuracy)
                runningLoss = 0.
    # store model results
    history = {
        'train_loss': arrTrainLoss,
        'train_accuracy': arrTrainAccuracy,
        'validation_loss': arrValidationLoss,
        'validation_accuracy': arrValidationAccuracy
    }
    return history


# Validate training with validation dataset. Used after each epoch.
def validateTraining(
    model: torch.nn.Module,
    device: torch.cuda.device,
    criterion: torch.nn.CrossEntropyLoss,
    validationLoader: torch.utils.data.DataLoader
):
    totalLoss = 0.
    correct = 0.
    VALIDATION_SIZE = len(validationLoader.dataset)
    # set model to evaluation mode
    model.eval()
    # loop through the validation dataset
    for _, (data, labels) in enumerate(validationLoader):
        # send validation data to device
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # get validation results
        output = model(data)
        # calculate training loss for this batch
        loss = criterion(output, labels)
        totalLoss += loss.item()
        # calculate validation accuracy
        output = torch.argmax(output, dim=1)
        correct += (output == labels).float().sum()
    # set model to train mode
    model.train()
    # calculate loss and accruacy
    loss = totalLoss / VALIDATION_SIZE
    accuracy = correct / VALIDATION_SIZE
    return loss, accuracy


# Defines the machine learning model layout.
# source: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # input layer
        self.input = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CHANNELS, out_channels=32, kernel_size=5, stride=1),
            torch.nn.ReLU()
        )
        # conv2d layer
        self.conv2d_1 = torch.nn.LazyConv2d(out_channels=32, kernel_size=5, stride=1, bias=False)
        self.conv2d_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(.25)
        )
        self.conv2d_3 = torch.nn.Sequential(
            torch.nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self.conv2d_4 = torch.nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, bias=False)
        self.conv2d_5 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(.25)
        )
        # flatten layer
        self.flatten = torch.nn.Flatten()
        # linear layer
        self.linear_1 = torch.nn.LazyLinear(out_features=256, bias=False)
        self.linear_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU()
        )
        self.linear_3 = torch.nn.LazyLinear(out_features=128, bias=False)
        self.linear_4 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU()
        )
        self.linear_5 = torch.nn.LazyLinear(out_features=84, bias=False)
        self.linear_6 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            torch.nn.Dropout(.25)
        )
        # output layer
        self.output = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=LABELS_TOTAL),
            torch.nn.Softmax(dim=1)
        )

    # Defines model layout.
    def forward(self, x):
        # input layer
        x = self.input(x)
        # conv2d layer
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        # flatten layer
        x = self.flatten(x)
        # linear layer
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)
        x = self.linear_5(x)
        x = self.linear_6(x)
        # output layer
        x = self.output(x)
        return x
