from util import *
from helpers import *
from sklearn.utils import shuffle
import pickle


# input
imgs_cars_file = './data/augmented/imgs_cars.p'
imgs_noncar_file = './data/augmented/imgs_noncar.p'

# output
X_scaler_file = './model/augmented/X_scaler.p'
X_train_file = './data/augmented/train/X_train.p'
y_train_file = './data/augmented/train/y_train.p'
X_test_file = './data/augmented/test/X_test.p'
y_test_file = './data/augmented/test/y_test.p'

# load imgs to array
with open(imgs_cars_file, 'rb') as file:
    car_list = pickle.load(file)

with open(imgs_noncar_file, 'rb') as file:
    noncar_list = pickle.load(file)


X_car = extract_features_from_img_list(car_list)

X_noncar = extract_features_from_img_list(noncar_list)

X_scaler, scaled_X = get_features_norm(X_car, X_noncar)

with open(X_scaler_file, 'wb') as file:
    pickle.dump(X_scaler, file)

print('number of feature inputs: ', len(scaled_X), 'feature shape: ', scaled_X[0].shape)

# construct labels
y = np.hstack((np.ones(len(car_list)), np.zeros(len(noncar_list))))
print('number of labels: ', len(y), 'label shape: ', y[0].shape)

# shuffle and splitting for training purpose
X, y = shuffle(scaled_X, y)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.10)

# save for later
with open(X_train_file, 'wb') as file:
    pickle.dump(X_train, file)

with open(y_train_file, 'wb') as file:
    pickle.dump(y_train, file)

with open(X_test_file, 'wb') as file:
    pickle.dump(X_test, file)

with open(y_test_file, 'wb') as file:
    pickle.dump(y_test, file)
