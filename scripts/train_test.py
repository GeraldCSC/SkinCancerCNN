import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import traintestloader 
import os
from nn import getmodel
wandb.init(project="skincancer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def geterror(vector, targetclass):
    output, i = vector.max(1)
    wrong = (i != targetclass).sum().item()
    return wrong 
def train(model,loader):
    numepoch = 300 
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss(reduction = 'sum') 
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
    model = getmodel().to(device)
    model = train(model,train_loader)
    testerror = test(model, test_loader)
    print(testerror)
