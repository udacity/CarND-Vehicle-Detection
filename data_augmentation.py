from helpers import *
import pickle


cars_path = '../vehicle_detect/vehicles/**/'
noncars_path = '../vehicle_detect/non-vehicles/**/'

imgs_cars_file = './data/raw/imgs_cars.p'
imgs_noncar_file = './data/raw/imgs_noncar.p'

imgs_cars_augmented_file = './data/augmented/imgs_cars.p'
imgs_noncar_augmented_file = './data/augmented/imgs_noncar.p'

# load imgs to array
car_list = get_img_array_from_path(cars_path, type='png')
noncar_list = get_img_array_from_path(noncars_path, type='png')
n_cars = len(car_list)
n_noncars = len(noncar_list)
print('n_cars: ', n_cars, 'n_noncars: ', n_noncars)

with open(imgs_cars_file, 'wb') as file:
    pickle.dump(car_list, file)

with open(imgs_noncar_file, 'wb') as file:
    pickle.dump(noncar_list, file)

# data augmentation
car_list_augmented = []
for i, car in enumerate(car_list):
    if i%1000 == 0:
        print('car augmentation progress: {}/{}.'.format(i, n_cars))
    car_list_augmented.append(car)
    n, new_imgs, = image_random_transform(car, prob=0.8)
    if n > 0:
        car_list_augmented.extend(new_imgs)

noncar_list_augmented = []
for i, noncar in enumerate(noncar_list):
    if i%1000 == 0:
        print('noncar augmentation progress: {}/{}.'.format(i, n_noncars))
    noncar_list_augmented.append(noncar)
    n, new_imgs, = image_random_transform(noncar, prob=0.8)
    if n > 0:
        noncar_list_augmented.extend(new_imgs)

with open(imgs_cars_augmented_file, 'wb') as file:
    pickle.dump(car_list_augmented, file)

with open(imgs_noncar_augmented_file, 'wb') as file:
    pickle.dump(noncar_list_augmented, file)

print('n_cars(augmented): ', len(car_list_augmented), 'n_noncars(augmented): ', len(noncar_list_augmented))
print('Done.')