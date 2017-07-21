import numpy as np
import cv2

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Lambda
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class CarClassifier:
    def __init__(self):
        self.model = None
        self.keras_callbacks = None

    def create_model(self, filename, input_shape=(64,64,3), num_classes = 2):
        self.model = Sequential()
        self.model.add(Lambda(lambda x : (x / 255.0) - 0.5, name='Normalize', input_shape=input_shape))
        self.model.add(Conv2D(3, 1, 1, border_mode='same', name='ColorConversion'))
        self.model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu', name='Conv1a'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name="MaxPool1"))
        self.model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu', name='Conv2a'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name="MaxPool2"))
        self.model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu', name='Conv3a'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name="MaxPool3"))
        self.model.add(Dropout(0.2, name='Dropout'))
        self.model.add(Flatten(name='Flatten'))
        self.model.add(Dense(4096, activation='relu', name='FC1'))
        self.model.add(Dense(1024, activation='relu', name='FC2'))
        self.model.add(Dense(num_classes, activation='softmax', name='FC3'))

        self.model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.keras_callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=1),
            ModelCheckpoint(filename, monitor='val_acc', save_best_only=True)]

    def train(self, train_generator, train_samples, val_generator, val_samples, epochs):
        self.model.fit_generator(train_generator, samples_per_epoch=train_samples,
                                 validation_data=val_generator,nb_val_samples=val_samples,
                                 nb_epoch=epochs,
                                 callbacks=self.keras_callbacks)

    def predict(self, imgs):
        pred = self.model.predict(np.asarray(imgs))
        return np.argmax(pred, axis=1)

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def restore(filename):
        clf = CarClassifier()
        clf.model = load_model(filename)

        return clf
