import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

df = pd.read_csv('/home/gerald/Desktop/MelanomaCNN/ISIC_2019_Training_GroundTruth.csv')
preprocess = transforms.Compose([
    transforms.ToTensor()])
def stratifyclass(df):
    imgname = df['image'].values 
    del df['image']
    oneofkmatrix = df.values 
    vectorofclasses = np.argmax(oneofkmatrix, axis = 1)
    train_name, test_name, train_label , test_label = train_test_split(imgname, vectorofclasses, 
                                    test_size= 0.30, random_state = 42, stratify = vectorofclasses)
    return train_name, train_label, test_name, test_label

#for now we wont consider a train val split, but only a test train split 
class SkinCancerDataSet(Dataset):
    def __init__(self, name, label, filepath):
        self.name = name
        self.label = label
        self.filepath = filepath
    def __getitem__(self, index):
        img = np.load(self.filepath +"/" + str(self.name[index]) + ".npy")    
        img = preprocess(img)
        y_truth = self.label[index].astype(int)
        return (img.float() , y_truth)

    def __len__(self):
        return len(self.name) 

def traintestloader(batch_size):
    train_name, train_label, test_name, test_label = stratifyclass(df)
    train_data = SkinCancerDataSet(train_name, train_label, 
                                   "/media/gerald/9ae8cbf0-f6c2-4065-8fb2-292211ac9a21/gerald/npytrain")
    test_data = SkinCancerDataSet(test_name, test_label, 
                                  "/media/gerald/9ae8cbf0-f6c2-4065-8fb2-292211ac9a21/gerald/npytrain")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle = True)
    return train_loader, test_loader

if __name__ == "__main__":
    pass
