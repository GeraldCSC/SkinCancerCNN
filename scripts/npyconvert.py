import numpy as np
import os
from preprocess import read_and_return_normalized_array
import math
dir_training = "/home/gerald/Desktop/MelanomaCNN/ISIC_2019_Training_Input"
dir_test = "/home/gerald/Desktop/MelanomaCNN/ISIC_2019_Test_Input"
npy_train = "/media/gerald/9ae8cbf0-f6c2-4065-8fb2-292211ac9a21/gerald/npytrain"
npy_test = "/media/gerald/9ae8cbf0-f6c2-4065-8fb2-292211ac9a21/gerald/npytest"
def padnow(matrix):
    if matrix.shape != (1024,1024,3):
        heighthalf = (1024 - matrix.shape[0]) / 2
        heightup = math.ceil(heighthalf)
        heightdown = math.floor(heighthalf)
        widthhalf = (1024 - matrix.shape[1]) / 2
        widthleft = math.ceil(widthhalf)
        widthright = math.floor(widthhalf) 
        paddedmatrix = np.pad(matrix, ((heightup, heightdown), (widthleft, widthright), (0,0)))
        return paddedmatrix
    else:
        return matrix

def converttonpy(incomingdir, outgoingdir):
    os.chdir(incomingdir)
    for root, dirs, files in os.walk("."):
        i = 0
        for item in files:
            if item.endswith(".jpg"):
                name = os.path.splitext(os.path.basename(item))[0]
                npyarray = read_and_return_normalized_array(os.path.abspath(item))
                npyarray = padnow(npyarray)
                assert npyarray.shape == (1024,1024,3)
                name = name + ".npy"
                np.save(outgoingdir + "/" + name, npyarray)
                i +=1
            print("converting file: " + str(i))
converttonpy(dir_training, npy_train)
converttonpy(dir_test, npy_test)

