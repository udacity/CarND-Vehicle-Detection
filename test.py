import cv2
from util import *

img_path = 'testing/4.jpg'

img = cv2.imread(img_path)
print('img shape: ', img.shape)

img_patch = img[200:232, 200:232]
print('img patch shape: ', img_patch.shape)

hog_subimg = hog_feature(img_patch, chan='ALL', pix_per_cell=8, cell_per_block=2, orient=8)

hog_unravel = hog_feature_unravel(img, chan='ALL', pix_per_cell=8, cell_per_block=2, orient=8)
hog_p = patch_features(img, p_size=32, hog_full=hog_unravel, xstart=200, ystart=200, hog_chan='ALL', color_hist_feat=False, spatial_feat=False)

print('subimg hog len: ', len(hog_subimg))

print(";;;;;;;;;;;;;;;;")

print('hog patch len', len(hog_p))

# print(hog_subimg)
print(";;;;;;;;;;;;;;;;")
# print(hog_p)

if np.equal(hog_subimg.any(), hog_p.any()):
    print("two hog's have equal element(s)")
else:
    print("NOT even a single element equal")

if np.allclose(hog_subimg, hog_p):
    print("YES")
else:
    print("NOT")
