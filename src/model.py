# internal libraries
from dictionary import (
    EPOCHS,
    BATCH_SIZE,
    CHANNELS,
    LABELS_TOTAL,
    GPU_DEVICE
)
from functionality import (
    getProgressbar,
    setProgressbarPrefix
)
# external libraries
import torch
import torch.utils.data


# Train model defined in the Model class.
def train(model: torch.nn.Module, device: torch.cuda.device, device_type: str, trainLoader: torch.utils.data.DataLoader, validationLoader: torch.utils.data.DataLoader):
    # branch if the device is set to GPU and send the model to the device
    if device_type is GPU_DEVICE:
        model.cuda(device)
    # set optimizer and criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    TRAIN_N = len(trainLoader.dataset)
    VALIDATION_N = len(validationLoader.dataset)
    # loop through each epoch
    for epoch in range(EPOCHS):
        train_correct = 0.
        train_loss_epoch = 0.
        train_running_loss = 0.
        # define the progressbar
        progressbar = getProgressbar(trainLoader, epoch, EPOCHS)
        # set model to training mode
        model.train()
        # loop through the training dataset
        for i, (data, labels) in enumerate(progressbar):
            # send training data to device
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # clear gradients
            optimizer.zero_grad()
            # get training results
            output = model(data)
            # compute gradients through backpropagation
            train_loss = criterion(output, labels)
            train_loss.backward()
            # apply gradients
            optimizer.step()
            # calculate running training loss
            train_running_loss += train_loss.item()
            train_loss_epoch += train_loss.item()
            # calculate training accuracy
            output = torch.argmax(output, dim=1)
            train_correct += (output == labels).float().sum()
            train_accuracy = 100 * train_correct / TRAIN_N
            # branch if batch size is reached to print more information
            if i >= (TRAIN_N / BATCH_SIZE) - 1:
                validation_correct = 0.
                validation_loss_epoch = 0.
                # set model to evaluation mode
                model.eval()
                # loop through the validation dataset
                for i, (data, labels) in enumerate(validationLoader):
                    # send validation data to device
                    data = data.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # get validation results
                    output = model(data)
                    # calculate training loss
                    validation_loss = criterion(output, labels)
                    validation_loss_epoch += validation_loss.item()
                    # calculate validation accuracy
                    output = torch.argmax(output, dim=1)
                    validation_correct += (output == labels).float().sum()
                # get parameters and set results in progressbar
                validation_accuracy = 100 * validation_correct / VALIDATION_N
                validation_loss_value = validation_loss_epoch / VALIDATION_N
                train_loss_value = train_loss_epoch / TRAIN_N
                setProgressbarPrefix(progressbar, train_loss_value, train_accuracy, validation_loss_value, validation_accuracy)
            elif i % BATCH_SIZE == (BATCH_SIZE - 1):
                train_loss_value = train_running_loss / (TRAIN_N / BATCH_SIZE)
                train_running_loss = 0.
                setProgressbarPrefix(progressbar, train_loss_value, train_accuracy)


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
