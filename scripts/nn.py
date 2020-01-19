import torch.nn as nn
class SkinNet(nn.Module):
    def __init__(self):
        super(SkinNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=1, padding = 1),
                                   nn.ReLU(), nn.MaxPool2d(4))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3, stride = 1, padding = 0),
                                   nn.ReLU(), nn.MaxPool2d(4))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128, 3, stride = 1, padding = 0), nn.ReLU(),
                                   nn.MaxPool2d(4))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256, 3, stride = 1, padding = 0), nn.ReLU(),
                                   nn.MaxPool2d(4))
        self.fc1 = nn.Sequential(nn.Linear(256 * 3 * 3, 200), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200,50), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(50,9), nn.Softmax(dim = 1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
def getmodel():
    return SkinNet()
