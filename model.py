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
print(len(trainX))
print(len(trainY))
print(len(image))
print(len(testX))
print(len(testY))
print(trainX.shape)

model = keras.models.Sequential([
    keras.layers.Dense(2048, input_shape=[2500, ], activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2048, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(43, activation='softmax')
])

model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=128, epochs = 4, validation_data=(testX, testY))

model.save("Model1")



