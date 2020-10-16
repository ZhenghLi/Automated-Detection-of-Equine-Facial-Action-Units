import torch
from torch import nn
import numpy as np


class Network(nn.Module):
    def __init__(self, class_number=1):
        super(Network, self).__init__()

        self.class_number = class_number

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # the region layer was replaced by a single 3*3 conv
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
        )

        self.avgpooling = nn.AdaptiveAvgPool2d((3,3))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*3*3, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=2048, out_features=class_number)
        )


    def forward(self, x):
        """

        :param x:   (b, c, h, w)
        :return:    (b, class_number)
        """
        x = self.extractor(x)
        x = self.avgpooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def drml(num_classes=1):
    model = Network(num_classes)
    return model

if __name__ == '__main__':
    from torch import nn
    from torch.autograd import Variable
    import numpy as np

    image = Variable(torch.randn(2, 3, 64, 64))
    label = Variable(torch.from_numpy(np.random.randint(2, size=[2, 1]))).float()

    net = drml(num_classes=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    while True:
        pred = net(image)
        pred = nn.Sigmoid()(pred)
        loss = nn.BCELoss(reduction='mean')(pred, label)
        print(loss.item())
        print('\n')
        opt.zero_grad()
        loss.backward()
        opt.step()

