#%% import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %% import data
train = pd.read_csv('train.csv')
print('train shape: ',train.shape)
train.head()

test = pd.read_csv('test.csv')
print('test shape: ',test.shape)
test.head()

#%% data splite Y_train and X_train
Y_train = train['label']
# print(Y_train)
X_train = train.drop(labels= ['label'], axis = 1)
# print(X_train)
#%% visualize digit classes
plt.figure(figsize = (15 ,7))
g = sns.countplot(Y_train, palette = 'pastel')
plt.title('Number of digit classes')
plt.show()
Y_train.value_counts()

#%% plote some sample
img = X_train.iloc[0].as_matrix()
img = img.reshape((28, 28))
plt.imshow(img, cmap = 'gray')
plt.title(train.iloc[0, 0])
plt.axis('off')
plt.show()


#%% plote some sample
img = X_train.iloc[6].as_matrix()
img = img.reshape((28, 28))
plt.imshow(img, cmap = 'gray')
plt.title(train.iloc[6, 0])
plt.axis('off')
plt.show()

#%% the data normalize
# print('X_train shape: ',X_train.shape)
# print('test shape: ',test.shape)
X_train = X_train / 255.0
test = test / 255.0
print('X_train shape: ',X_train.shape)
print('test shape: ',test.shape)


#%% reshape
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
print('X_train shape: ',X_train.shape)
print('test shape: ',test.shape)


#%% label encoding
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)
# print(Y_train)
#%% the data train and test splite
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)
#%% some examples
plt.imshow(X_train[3][:, :, 0], cmap = 'gray')
plt.show()
#%% import keras
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


#%% create model
model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.25))

# fully connected

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))


#%% adam optimaizer
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
#%% compile the model
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%% epochs and batch size
epochs = 3
batch_size = 250


#%% data augmentaion
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0.5,
    zoom_range = 0.5,
    width_shift_range = 0.5,
    height_shift_range = 0.5,
    horizontal_flip = False,
    vertical_flip = False
)
datagen.fit(X_train)


#%% fit model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), epochs = epochs, validation_data = (X_val, Y_val), steps_per_epoch = X_train.shape[0] // batch_size)
#%% plot the loss and accuracy
plt.plot(history.history['val_loss'], color = 'b', label = 'val loss')
plt.title('Test Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#%% confusion matrix
Y_predict = model.predict(X_val)
Y_predict_classes = np.argmax(Y_predict, axis = 1)
Y_true = np.argmax(Y_val, axis = 1)
confusion_mtx = confusion_matrix(Y_true, Y_predict_classes)

f, ax = plt.subplot(figsize = (8, 8)) 
sns.heatmap(confusion_mtx, annot = True, linewidths = 0.01, cmap = 'greens', linecolor = 'gray', fmt = '.1f', ax = ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


#%%
