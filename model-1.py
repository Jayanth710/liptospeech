# 0_1. Load keras package needed
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.datasets import mnist
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# 0_2. Fix random seed
np.random.seed(3)

# 1_1. Creating Datasets
# Load MNIST dataset from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print ( "Initial train data shape is",x_train.shape, "Initial train label shape is", y_train.shape)

# Reshape dataset to model input shape 
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
print ( "Converted train data shape is",x_train.shape)
# Reshape label to one-hat shape for classification problem 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print ("Converted train label shape is", y_train.shape)

# 2_1. Buliding a Model
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu')) # First layer with input shape
model.add(Dense(units=10, activation='softmax')) # Second layer with sofrmax activation function

# Model Architecture Visualization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True))

# 1_2. Creating Datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print ( "Initial train data shape is",x_train.shape, "Initial train label shape is", y_train.shape)

# Reshape dataset to model input shape 
x_train = x_train.reshape(60000, 28, 28,1).astype('float32') / 255.0
x_test = x_test.reshape(10000, 28, 28,1).astype('float32') / 255.0
print ( "Converted train data shape is",x_train.shape)
# Reshape label to one-hat shape for classification problem 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print ("Converted train label shape is", y_train.shape)

# 2_2. Buliding a Model with Convolutional layer
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))  # First Convolutional layer with input shape 
model.add(Conv2D(64, (3, 3), activation='relu')) # Second Convolutional layer 
model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling layer
model.add(Flatten())  # Flatten layer for converting CNN codes to Dense layer input 
model.add(Dense(128, activation='relu')) # hidden Dense layer
model.add(Dense(10, activation='softmax')) # Activation layer with sofrmax

# Model Architecture Visualization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True))

# 3. Setting up the Model Learning Process
# Define optimizer for training
adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compiling sequential model with optimizer
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 4. Training the Model
hist = model.fit(x_train, y_train, batch_size=32, validation_split=0.2, shuffle=True, epochs=10)

# 5. Confirm the Learning Process
# by mathplot model training history can be visualized
# %matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 6. Evaluating the Model
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('evaluation loss and acc')
print(loss_and_metrics)

# 7. Using the Model

from keras.models import load_model

model.save('mnist_mlp_model.h5')

from keras.models import load_model

model = load_model('mnist_mlp_model.h5')

xhat = x_test[0:1]
print(y_test[0:1])
yhat = (model.predict(xhat) > 0.5).astype("int32")
# yhat = model.predict_classes(xhat)
yhatp = model.predict(xhat)
print('predicted class')
print(yhat)
print('class probabilty is')
print(yhatp)