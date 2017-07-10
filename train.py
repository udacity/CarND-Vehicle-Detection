import numpy as np
import cv2

import glob
import os.path
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from car_classifier import CarClassifier

NOT_CAR = 0
CAR = 1

class TrainClassifier: 
    def __init__(self, classifier):
        self.filenames = [[], []]
        self.features  = [[], []]

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.classifier = classifier

    def load_images(self, dir, type):
        for filename in glob.glob(os.path.join(dir, '**', '*.png'), recursive=True):
            self.filenames[type].append(filename)

    def extract_features(self, type):
        for filename in tqdm(self.filenames[type]):
            # load image
            img = cv2.imread(filename)

            # prepare image
            feature_img = self.classifier.prepare_img(img)

            # extract features
            features = []

            # -- hog features 
            features.extend(self.classifier.extract_hog_features(feature_img))

            # -- color features
            features.extend(self.classifier.extract_color_features(feature_img))

            # store features
            self.features[type].append(np.ravel(features))

    def create_train_test_sets(self):
        # stack the features arrays together in a numpy array
        X = np.vstack((self.features[CAR], self.features[NOT_CAR])).astype(np.float64)                        

        # normalize the features    
        self.classifier.scaler_initialize(X)
        X_norm = self.classifier.scaler_apply(X)

        # create the labels array
        y = np.hstack((
            np.ones(len(self.features[CAR])), 
            np.zeros(len(self.features[NOT_CAR]))
            ))

        # randomize and split the data into training and test datasets
        random_state = np.random.randint(0, 100)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_norm, y, test_size=0.2, random_state=random_state)

    def train_classifier(self):
        # fit the classifier to the training dataset
        self.classifier.classifier_train(self.X_train, self.y_train)

        # print the accuracy of the classifier
        score = self.classifier.classifier_accuracy(self.X_test, self.y_test)
        print("Classifier accuracy = {:.4f}".format(score))

if __name__ == '__main__':
    clf = CarClassifier()
    tc = TrainClassifier(clf)

    print('Loading images ... ', end='')
    tc.load_images(os.path.join('data', 'vehicles'), CAR)
    tc.load_images(os.path.join('data', 'non-vehicles'), NOT_CAR)
    print('done')

    print('Extracting features ...')
    tc.extract_features(CAR)
    tc.extract_features(NOT_CAR)

    print('Preparing data sets ...', end='')
    tc.create_train_test_sets()
    print('done')

    print('Training the classifier ...')
    tc.train_classifier()

    print('Saving the classifier')
    clf.save('classifier_svc.pkl')