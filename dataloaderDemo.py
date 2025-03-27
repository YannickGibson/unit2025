import numpy as np
from matplotlib import pyplot as plt
import pprint

def dataloaderdemo(x,y,n,tstart,tend,c):
    img = np.zeros([x,y,c],dtype=np.uint8)
    imgRange = np.repeat(np.array([img]), tend-tstart, 0)
    imgArr = np.repeat(np.array([imgRange]), n, 0)
    return imgArr

#print(dataloaderdemo(100, 100, 1, 0, 5, 3)[0][0].shape)
plt.imshow(dataloaderdemo(100, 100, 2, 0, 5, 3)[0][0], interpolation='nearest')
