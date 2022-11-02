"""
Massimo Zimmerman
Dataset Classification Project
PRE-PLAN
"""

# Importing/Activating Libraries for Project 2
import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


"""
PLAN 1
"""

"""
The Data is the CIFAR10 small images classification dataset provided by 'tf.keras.dataset', 
which is a dataset of 50,000 32x32 color training images and 10,000 test images,
labeled over 10 classes, containing 6000 images each.
"""
#------------------------------------------------------------------------------------------------
# Allocate/Assign the training and testing sets pre-defined by the CIRFAR100 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshaping the Images from the CIFAR100 Dataset
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

# Data Pro-processing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Pool the classified data to randomize and resize the training/testing data, as CIFAR10 presets the
# Training size to 50,000 and Testing Size to 10,000
raw_data = np.concatenate((x_train, x_test), axis = 0)
labels = np.concatenate((y_train, y_test), axis = 0) 

x_train, x_test, y_train, y_test = train_test_split(raw_data, labels, train_size = 0.8, shuffle = True, random_state = 77)

print(x_train.shape)
print(x_test.shape)

# Transform our labels into a one-hot encoding 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


"""
PLAN 2
"""
#------------------------------------------------------------------------------------------------
# Modeling the Convolutional Neural Network

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=x_train[0,:,:,:].shape),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(rate=0.25),
    
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(rate=0.25),
    
    Conv2D(filters=192, kernel_size=(3,3), activation='relu'),
    Conv2D(filters=192, kernel_size=(3,3), activation='relu'),
        
    Flatten(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(rate=0.5),
    Dense(10, activation='softmax')
])
model.summary()

"""
PLAN 3
"""
#------------------------------------------------------------------------------------------------
# Compile the Model to calculate Accuracy, Loss, & the Confusion Matrix
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# Training and fitting the model 
epochs = 5
batch_size = 32
history = model.fit(
    x_train, 
    y_train, 
    batch_size=batch_size, 
    epochs=epochs,
    validation_data=(x_test, y_test))


# Visualizing and analyzing the training/testing accuracy and loss for each epoch
plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Computes the Confusion Matrix for the Dataset
predictions = model.predict(x_test)
y_pred = (predictions > 0.5)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

plt.figure()
plt.imshow(matrix, cmap = 'gray')
plt.title('Confusion Matrix')
