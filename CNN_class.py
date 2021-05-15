import os
import csv
import tensorflow as tf  # 2.0
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, Dense, Dropout, Activation, Flatten
from keras.layers import Input, concatenate
from keras import optimizers
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from augmentedvolumetricimagegenerator.generator import customImageDataGenerator
from keras.callbacks import EarlyStopping


# Administrative items
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Where the file is located
path = r'C:\Users\jesse\OneDrive\Desktop\Research\PD\decline'
folder = os.listdir(path)

target_size = (96, 96, 96)


# creating x - converting images to array
def read_image(path, folder):
    mri = []
    for i in range(len(folder)):
        files = os.listdir(path + '\\' + folder[i])
        for j in range(len(files)):
            image = np.array(nib.load(path + '\\' + folder[i] + '\\' + files[j]).get_fdata())
            image = np.resize(image, target_size)
            image = np.expand_dims(image, axis=3)
            image /= 255.
            mri.append(image)
    return mri

# creating y - one hot encoder
def create_y():
    excel_file = r'C:\Users\jesse\OneDrive\Desktop\Research\PD\decline_label.xlsx'
    excel_read = pd.read_excel(excel_file)
    excel_array = np.array(excel_read['Label'])
    label = LabelEncoder().fit_transform(excel_array)
    label = label.reshape(len(label), 1)
    onehot = OneHotEncoder(sparse=False).fit_transform(label)
    return onehot

# Splitting image train/test
x = np.asarray(read_image(path, folder))
y = np.asarray(create_y())
x_split, x_test, y_split, y_test = train_test_split(x, y, test_size=.2, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_split, y_split, test_size=.25, stratify=y_split)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)


batch_size = 10
num_classes = len(folder)

inputs = Input((96, 96, 96, 1))
conv1 = Conv3D(32, [3, 3, 3], padding='same', activation='relu')(inputs)
conv1 = Conv3D(32, [3, 3, 3], padding='same', activation='relu')(conv1)
pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)
#drop1 = Dropout(0.5)(pool1)

conv2 = Conv3D(64, [3, 3, 3], padding='same', activation='relu')(pool1)
conv2 = Conv3D(64, [3, 3, 3], padding='same', activation='relu')(conv2)
pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
#drop2 = Dropout(0.5)(pool2)

conv3 = Conv3D(128, [3, 3, 3], padding='same', activation='relu')(pool2)
conv3 = Conv3D(128, [3, 3, 3], padding='same', activation='relu')(conv3)
pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
#drop3 = Dropout(0.5)(pool3)

flat1 = Flatten()(pool3)
dense1 = Dense(128, activation='relu')(flat1)
#drop5 = Dropout(0.5)(dense1)
dense2 = Dense(num_classes, activation='sigmoid')(dense1)

model = Model(inputs=[inputs], outputs=[dense2])

opt = optimizers.Adam(lr=1e-4)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(score)


y_pred = model.predict(x_test, batch_size=batch_size)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
confusion = confusion_matrix(y_test, y_pred)
map = sns.heatmap(confusion, annot=True)
print(map)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(1)
plt.plot(acc)
plt.plot(val_acc)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.title('Accuracy')

plt.figure(2)
plt.plot(loss)
plt.plot(val_loss)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.title('Loss')