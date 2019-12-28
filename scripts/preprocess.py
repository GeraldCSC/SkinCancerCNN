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
    arrayr = exposure.equalize_adapthist(matrix[:,:,0])
    arrayg = exposure.equalize_adapthist(matrix[:,:,1])
    arrayb = exposure.equalize_adapthist(matrix[:,:,2])
    array1 = np.dstack((arrayr,arrayg,arrayb))
    return array1
def read_and_return_normalized_array(filepath):
    array = return_array(filepath)
    normalized_array = equalize_adaptivehistogram(array)
    return normalized_array
