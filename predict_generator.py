import numpy as np
import keras
from augmentedvolumetricimagegenerator.generator import customImageDataGenerator


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, images, scores, batch_size=32, dim=(96, 96, 96), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.images = images
        self.scores = scores
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.scores) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate a single batch
        x_batch = [self.images[k] for k in batch_index]
        y_batch = [self.scores[k] for k in batch_index]

        # Generate data
        x, y = self.__data_generation(x_batch, y_batch)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.scores))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_batch, y_batch):
        'Generates data containing batch_size samples'  # x : (n_samples, *dim, n_channels)
        
#         datagen = customImageDataGenerator(shear_range=0.2,
#                                            zoom_range=0.2,
#                                            horizontal_flip=True)
#         dataset = datagen.flow(x_batch, y_batch, batch_size=self.batch_size)
        
        # Generate data
        for i in range(len(x_batch)):
            x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
            y = np.empty((self.batch_size), dtype=float)

            x[i] = x_batch[i]
            y[i] = y_batch[i]

        return x, y