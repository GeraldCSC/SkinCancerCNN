import cv2
import numpy as np
from skimage import exposure
def return_array(filepath):
    im = cv2.imread(filepath)
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    return im2
def equalize_adaptivehistogram(matrix):
    arrayr = exposure.equalize_adapthist(array[:,:,0])
    arrayg = exposure.equalize_adapthist(array[:,:,1])
    arrayb = exposure.equalize_adapthist(array[:,:,2])
    array1 = np.dstack((arrayr,arrayg,arrayb))
    return array1
array = return_array("ISIC_2019_Training_Input/ISIC_0026424.jpg")
normalized_array = equalize_adaptivehistogram(array)
import matplotlib.pyplot as plt
plt.imshow(normalized_array)
plt.show()
