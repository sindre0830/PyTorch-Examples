# internal libraries
from dictionary import (
    EPOCHS,
    BATCH_SIZE,
    CHANNELS,
    LABELS_TOTAL
)
# external libraries
import torch
import torch.utils.data
import tqdm


# Train model defined in the Model class.
def train(model: torch.nn.Module, device: torch.cuda.device, trainDatasetLoader: torch.utils.data.DataLoader):
    if torch.cuda.is_available():
        model.cuda(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    N = len(trainDatasetLoader.dataset)
    for epoch in range(EPOCHS):
        correct = 0
        progressbar = tqdm.tqdm(
            iterable=trainDatasetLoader,
            desc='Epoch {:>2}/{}'.format(epoch + 1, EPOCHS),
            ncols=150,
            ascii='░▒',
            unit=' step'
        )
        for i, (data, labels) in enumerate(progressbar):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            output = torch.argmax(output, dim=1)
            correct += (output == labels).float().sum()
            accuracy = 100 * correct / N
            if i % BATCH_SIZE == (BATCH_SIZE - 1):
                progressbar.set_postfix_str("Loss: {:.4f}, Accuracy: {:.4f}".format(loss.item(), accuracy))
    model.train(False)


# Defines the machine learning model layout.
# source: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CHANNELS, out_channels=32, kernel_size=5, stride=1),
            torch.nn.ReLU()
        )
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
        self.flatten = torch.nn.Flatten()
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
        self.output = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=LABELS_TOTAL),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)
        x = self.linear_5(x)
        x = self.linear_6(x)
        x = self.output(x)
        return x
