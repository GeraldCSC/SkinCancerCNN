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
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.unsqueeze(0)
def geterror(vector, targetclass):
    output, indices = vector.max(1) 
    if targetclass == indices:
        return 0
    else:
        return 1 
def train(loader):
    numepoch = 5
    model = SkinNet().to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01)
    model.train()
    for i in range(numepoch):
        losslist = []
        errorlist = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            optimizer.zero_grad()
            loss = criterion(y_pred, y) 
            loss.backward()
            optimizer.step()
            losslist.append(loss.item())
            errorlist.append(geterror(y_pred, y))
            count += 1
        trainloss = sum(losslist) / len(losslist)
        trainerror = sum(errorlist) /len(errorlist)
        wandb.log({"Epoch": i, "Training Loss": trainloss, "Training Error": trainerror}) 
        print("Epoch: " + str(i) + " Training Loss: " +str(trainloss) + " Training Error" + str(trainerror))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    torch.save(model.state_dict(), "skinmodel.pt")
    return model
def test(model, loader):
    model.eval() 
    errorlist = [] 
    for x , y in loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x) 
            errorlist.append(geterror(y_pred,y))         
            count += 1
    return sum(errorlist) /len(errorlist)
if __name__ == "__main__":
    batchsize = 1
    train_loader , test_loader = traintestloader(batchsize)
    model = train(train_loader)
    testerror = test(model, test_loader)
    f= open("/home/gerald/Desktop/MelanomaCNN/test_accuracy.txt","a")
    f.write(str(testerror))
    f.close()
