import numpy as np
import cv2

import glob
import os.path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from car_classifier import CarClassifier

NOT_CAR = 0
CAR = 1

class TrainClassifier: 
    def __init__(self, classifier):
        self.filenames = [[], []]

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.train_gen = None
        self.test_gen = None

        self.classifier = classifier

    def load_images(self, dir, type):
        for filename in glob.glob(os.path.join(dir, '**', '*.png'), recursive=True):
            self.filenames[type].append(filename)

    def batch_generator(self, X, y, batch_size=64):
        n_samples = len(X)

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.uint8)
        y_batch = np.zeros((batch_size, 2), dtype=np.int32)

        while True:
            X, y = shuffle(X, y)

            for batch_i in range(0, n_samples, batch_size):
                n = min(batch_size, n_samples - batch_i)
                for idx in range(0, n):
                    # load image
                    X_batch[idx] = cv2.imread(X[batch_i+idx])
                    y_batch[idx] = y[batch_i+idx]
                yield (X_batch[0:n], y_batch[0:n])

    def create_train_test_sets(self):
        # stack the filename arrays together in a numpy array
        X = np.array(self.filenames[CAR] + self.filenames[NOT_CAR])

        # create the labels array
        n1 = (len(self.filenames[CAR]), 1)
        n2 = (len(self.filenames[NOT_CAR]), 1)

        y = np.vstack((
            np.hstack((np.ones(n1), np.zeros(n1))),
            np.hstack((np.zeros(n2), np.ones(n2)))
            ))

        # randomize and split the data into training and test datasets
        random_state = np.random.randint(0, 100)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        self.train_gen = self.batch_generator(self.X_train, self.y_train)
        self.test_gen = self.batch_generator(self.X_test, self.y_test)

    def train_classifier(self):
        self.classifier.create_model('classifier.h5')
        self.classifier.train(self.train_gen, len(self.X_train), self.test_gen, len(self.X_test), 50)


if __name__ == '__main__':
    clf = CarClassifier()
    tc = TrainClassifier(clf)

    print('Loading images ... ', end='')
    tc.load_images(os.path.join('data', 'vehicles'), CAR)
    tc.load_images(os.path.join('data', 'non-vehicles'), NOT_CAR)
    print('done')

    print('Preparing data sets ...', end='')
    tc.create_train_test_sets()
    print('done')

    print('Training the classifier ...')
    tc.train_classifier()