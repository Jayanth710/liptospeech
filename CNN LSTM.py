# 0. Load keras package needed
import numpy as np
import tensorflow as tf
import keras
import os # drectory library
import cv2 # image processing library
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
# Fix random seed
np.random.seed(3)

timesteps = 10 # input frame numbers for LSTM
n_labels = 8 # Number of Dataset Labels
Learning_rate = 0.0001 # Oprimizers lr, in this case, for adam
batch_size = 32
validation_ratio = 0.2 
num_epochs = 50
img_col = 128 # Transfer model input size ( MobileNet )
img_row = 128 # Transfer model input size ( MobileNet )
img_channel = 3 # RGB

# 1. Creating Datasets
# define temporary empty list for load
data = []
label = []
Totalnb = 0

# Load Dataset
for i in range(n_labels):
    nb = 0
    # Counting datasets in each labels
    for root, dirs, files in os.walk('normalized cascade/' + str(i+1)): # set directory
        for name in dirs:
            nb = nb + 1
    print(i,"Label number of Dataset is:",nb)
    Totalnb = Totalnb + nb
    # by Counting size, cross subfolder and read image data, resize image, and append list 
    for j in range(nb):
        temp = []
        for k in range(timesteps):
            name = 'normalized cascade/' + str(i+1) + '/' + str(j+1) + '/' + str(k+1) + '.jpg'
            img = cv2.imread(name)
            res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
            temp.append(res)
        label.append(i)        
        data.append(temp)
print("Total Number of Data is",Totalnb)

# Convert List to numpy array, for Keras use
Train_label = np.eye(n_labels)[label] # One-hot encoding by np array function
Train_data = np.array(data)
print("Dataset shape is",Train_data.shape, "(size, timestep, column, row, channel)")
print("Label shape is",Train_label.shape,"(size, label onehot vector)")

# shuffling dataset for input fit function
# if don`t, can`t train model entirely
x = np.arange(Train_label.shape[0])
np.random.shuffle(x)
# same order shuffle is needed
Train_label = Train_label[x]
Train_data = Train_data[x]

# declare data for training and validation, if you want, you can seperate testset from this
X_train=Train_data[0:Totalnb,:]
Y_train=Train_label[0:Totalnb]

# 2. Buliding a Model
# declare input layer for CNN+LSTM architecture
video = Input(shape=(timesteps,img_col,img_row,img_channel))
# Load transfer learning model that you want
model = MobileNet(input_shape=(img_col,img_row,img_channel), weights="imagenet", include_top=False)
model.trainable = False
# FC Dense Layer
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.3)(x)
cnn_out = Dense(128, activation="relu")(x)
# Construct CNN model 
Lstm_inp = Model(model.input, cnn_out)
# Distribute CNN output by timesteps 
encoded_frames = TimeDistributed(Lstm_inp)(video)
# Contruct LSTM model 
encoded_sequence = LSTM(256)(encoded_frames)
hidden_Drop = Dropout(0.3)(encoded_sequence)
hidden_layer = Dense(128, activation="relu")(encoded_sequence)
outputs = Dense(n_labels, activation="softmax")(hidden_layer)
# Contruct CNN+LSTM model 
model = Model([video], outputs)
# 3. Setting up the Model Learning Process
# Model Compile 
adam = tf.keras.optimizers.Adam(lr=Learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# 4. Training the Model
hist = model.fit(X_train, Y_train, batch_size=batch_size, validation_split=validation_ratio, shuffle=True, epochs=num_epochs)

# Model Architecture Visualization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True))

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# 5. Confirm the Learning Process
# %matplotlib inline 
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')  
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')  
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 6. Using the Model
model.save('Lib_Reading_10Frame_Model.h5')