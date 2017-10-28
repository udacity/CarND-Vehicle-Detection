from util import *
from helpers import *
from sklearn.utils import shuffle
import pickle


cars_path = '../vehicle_detect/vehicles_smallset/**/'
noncars_path = '../vehicle_detect/non-vehicles_smallset/**/'
X_scaler_file = './model/X_scaler_small.p'
X_train_file = './data/X_train_small.p'
y_train_file = './data/y_train_small.p'

# load imgs to array
car_list = get_img_array_from_path(cars_path, type='jpeg')
print('n_cars: ', len(car_list))
noncar_list = get_img_array_from_path(noncars_path, type='jpeg')
print('n_noncars: ', len(noncar_list))


X_car = extract_features_from_img_list(car_list, color_space='HSV', spatial_size=(32, 32),
                                       color_bins=32, orient=32,
                                       pix_per_cell=16, cell_per_block=2, hog_chan=0,
                                       spatial_feat=True, color_hist_feat=True, hog_feat=True)

X_noncar = extract_features_from_img_list(noncar_list, color_space='HSV', spatial_size=(32, 32),
                                          color_bins=32, orient=32,
                                          pix_per_cell=16, cell_per_block=2, hog_chan=0,
                                          spatial_feat=True, color_hist_feat=True, hog_feat=True)


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



