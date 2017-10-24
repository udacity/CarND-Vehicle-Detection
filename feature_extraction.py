from util import *
from helpers import *
from sklearn.utils import shuffle
import pickle


cars_path = '../vehicle_detect/vehicles/**/'
noncars_path = '../vehicle_detect/non-vehicles/**/'


# load imgs to array
car_list = get_img_array_from_path(cars_path, type='png')
print('n_cars: ', len(car_list))
noncar_list = get_img_array_from_path(noncars_path, type='png')
print('n_noncars: ', len(noncar_list))


X_car = extract_features_from_img_list(car_list, color_space='HLS', orient=9, pix_per_cell=8, hist_bins=32, hog_channel='ALL', spatial_feat=False, spatial_size=(16,16))

X_noncar = extract_features_from_img_list(noncar_list, color_space='HLS', orient=9, pix_per_cell=8, hist_bins=32, hog_channel='ALL', spatial_feat=False, spatial_size=(16,16))

with open('X_car.p', 'wb') as file:
    pickle.dump(X_car, file)


with open('X_noncar.p', 'wb') as file:
    pickle.dump(X_noncar, file)


X_scaler, scaled_X = get_features_norm(X_car, X_noncar)

with open('X_scaler.p', 'wb') as file:
    pickle.dump(X_scaler, file)


print('num feature inputs: ', len(scaled_X), 'feature shape: ', scaled_X[0].shape)

# construct labels
y = np.hstack((np.ones(len(car_list)), np.zeros(len(noncar_list))))
print('labels: ', len(y))

# shuffle and splitting for training purpose
X, y = shuffle(scaled_X, y)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.10)

# save for later
with open('X_train.p', 'wb') as file:
    pickle.dump(X_train, file)

with open('y_train.p', 'wb') as file:
    pickle.dump(y_train, file)

with open('X_test.p', 'wb') as file:
    pickle.dump(X_test, file)

with open('y_test.p', 'wb') as file:
    pickle.dump(y_test, file)
