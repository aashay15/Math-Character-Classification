import PIL
from PIL import Image
from skimage.transform import resize
from scipy import misc
from scipy import ndimage
import os
#import pillow
import numpy as np
from keras import models
image = PIL.Image.open("3.png")
img = image.convert(mode="1", dither = Image.NONE)
print(img.size)
img2 = img.resize((50,50))
#img2.show()
print(img2.size)
img2 = np.asarray(img2, dtype=int)
img3 = img2.reshape(-1)
print(img3.shape)
print(type(img3))
#img3 = img2.flatten()
#print(img3.shape)
#print(len(img3))
##print(type(img3))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras


file = open('outputs/train/train.pickle', 'rb')

image = pickle.load(file)
#print(image)
trainX = []
trainY = []
testX = []
testY = []
for i in range(0,(len(image)-10000)):
    trainX.append(image[i]['features'])
    trainY.append(image[i]['label'])

for i in range(61378, 71378):
    testX.append(image[i]['features'])
    testY.append(image[i]['label'])
file.close()
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

model = models.load_model("Model1")
prediction = model.predict(np.array([img3]))
prediciton2 = model.predict(np.array([testX[200]]))
print(np.exp(prediction))
print(np.exp(prediciton2))
print(testY[200])