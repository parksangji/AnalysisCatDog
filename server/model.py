from pickletools import optimize
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

def preprocess(img,label):
    return tf.image.resize(img,[HEIGHT,WIDTH])/255,label

className = ['cat','dog']
split = ['train[:70%]','train[70%:]']

trainDataset, testDataset = tfds.load(name='cats_vs_dogs',split=split,as_supervised= True)

HEIGHT = 200
WIDTH = 200

trainDataset = trainDataset.map(preprocess).batch(32)
testDataset = testDataset.map(preprocess).batch(32)

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

trainHistory = model.fit(trainDataset,epochs=10,validation_data =testDataset)

trainHistory.save('catdogModel')