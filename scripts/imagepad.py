import os
import numpy as np
import math
npyfilepath = "/media/gerald/9ae8cbf0-f6c2-4065-8fb2-292211ac9a21/gerald/npytrain"
def pad():
    i = 0
    filetorewrite = []
    for root, dirs, files in os.walk(npyfilepath):
        os.chdir(npyfilepath)
        for item in files:
            if item.endswith(".npy"):
                matrix = np.load(item)
                if matrix.shape != (1024,1024,3):
                    heighthalf = (1024 - matrix.shape[0]) / 2
                    heightup = math.ceil(heighthalf)
                    heightdown = math.floor(heighthalf)
                    widthhalf = (1024 - matrix.shape[1]) / 2
                    widthleft = math.ceil(widthhalf)
                    widthright = math.floor(widthhalf) 
                    paddedmatrix = np.pad(matrix, ((heightup, heightdown), (widthleft, widthright), (0,0)))
                    if paddedmatrix.shape != (1024,1024,3):
                        print("ERROR IN FILE: " + item)
                        print("SHAPE OF ITEM IS: " + str(matrix.shape))
                        print("Padded matrix is of shape: " + str(paddedmatrix.shape))
                        filetorewrite.append(item)
                    else:
                        np.save(item, paddedmatrix)
                        i += 1
                        print("converting file number: " + str(i))
            break
    f = open("/home/gerald/Desktop/MelanomaCNN/pythonscripts/filetorewrite.txt", "a")
    f.write(str(filetorewrite))
    f.close()

