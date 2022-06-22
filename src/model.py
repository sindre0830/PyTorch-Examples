# external libraries
import torch


# Defines the machine learning model layout.
# source: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            torch.nn.ReLU(inplace=True)
        )
        self.conv2d_1 = torch.nn.LazyConv2d(out_channels=32, kernel_size=5, stride=1, bias=False)
        self.conv2d_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(.25, inplace=True)
        )
        self.conv2d_3 = torch.nn.Sequential(
            torch.nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2d_4 = torch.nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, bias=False)
        self.conv2d_5 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(.25, inplace=True)
        )
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.LazyLinear(out_features=256, bias=False)
        self.linear_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(inplace=True)
        )
        self.linear_3 = torch.nn.LazyLinear(out_features=128, bias=False)
        self.linear_4 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(inplace=True)
        )
        self.linear_5 = torch.nn.LazyLinear(out_features=84, bias=False)
        self.linear_6 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(.25, inplace=True)
        )
        self.output = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=10),
            torch.nn.Softmax()
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
