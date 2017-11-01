from util import *
from helpers import *
from sklearn.utils import shuffle
import pickle


cars_path = '../vehicle_detect/vehicles_smallset/**/'
noncars_path = '../vehicle_detect/non-vehicles_smallset/**/'
X_scaler_file = './model/raw/X_scaler_small.p'
X_train_file = './data/raw/X_train_small.p'
y_train_file = './data/raw/y_train_small.p'

# load imgs to array
car_list = get_img_array_from_path(cars_path, type='jpeg')
print('n_cars: ', len(car_list))
noncar_list = get_img_array_from_path(noncars_path, type='jpeg')
print('n_noncars: ', len(noncar_list))


X_car = extract_features_from_img_list(car_list)

X_noncar = extract_features_from_img_list(noncar_list)


X_scaler, scaled_X = get_features_norm(X_car, X_noncar)

print('num feature inputs: ', len(scaled_X), 'feature shape: ', scaled_X[0].shape)

# construct labels
y = np.hstack((np.ones(len(car_list)), np.zeros(len(noncar_list))))
print('labels: ', len(y))

# shuffle and splitting for training purpose
X, y = shuffle(scaled_X, y)

# save for later
with open(X_scaler_file, 'wb') as file:
    pickle.dump(X_scaler, file)

# save for later
with open(X_train_file, 'wb') as file:
    pickle.dump(X, file)

with open(y_train_file, 'wb') as file:
    pickle.dump(y, file)



