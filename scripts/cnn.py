import wandb
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import traintestloader 
import os

wandb.init(project="skincancer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.fc1 = nn.Sequential(nn.Linear(256 * 3 * 3, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100,50), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(50,9), nn.Softmax(dim = 0))
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
def geterror(vector, targetclass):
    output, i = vector.max(1)
    wrong = (i != targetclass).sum().item()
    return wrong 
def train(loader):
    numepoch = 200 
    model = SkinNet().to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    for i in range(numepoch):
        count = 0
        numdata = 0
        error = 0
        lossvar = 0
        for x, y in loader:
            numdata += len(y)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y) 
            loss.backward()
            optimizer.step()
            lossvar += loss.item() 
            error += geterror(y_pred,y)
            count +=1
            if count == 200:
                count = 0
                wandb.log({"Training Loss per 10k": loss.item(), 
                           "Training Error per 10k": (error/numdata)}) 
        trainloss = lossvar / numdata 
        trainerror = error / numdata
        wandb.log({"Epoch": i, "Training Loss": trainloss, 
                   "Training Error": trainerror}) 
        print("Epoch: " + str(i) + " Training Loss: " +
              str(trainloss) + " Training Error: " + str(trainerror))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    torch.save(model.state_dict(), "skinmodel.pt")
    return model
def test(model, loader):
    model.eval() 
    numdata = 0
    error = 0
    count = 0
    for x , y in loader:
        count += 1
        with torch.no_grad():
            numdata += len(y)
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x) 
            error += geterror(y_pred,y)
    return error / numdata
if __name__ == "__main__":
    batchsize = 10 
    train_loader , test_loader = traintestloader(batchsize)
    model = train(train_loader)
    testerror = test(model, test_loader)
    f= open("/home/gerald/Desktop/MelanomaCNN/test_accuracy.txt","a")
    f.write(str(testerror))
    f.close()
