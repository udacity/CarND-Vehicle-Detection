import numpy as np
import cv2

import json
import glob
import os.path

class Camera:
    def __init__(self, name='camera'):
        self.name = name
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def calibrate(self, path_mask, grid_x, grid_y):
        # try loading the results from a previous calibration
        if self.load():
            return True
        
        # calibrate the camera
        print('Calibrating camera using images from ' + path_mask)
        
        objpoints = []
        imgpoints = []
        img_shape = None

        # prepare object points
        objp = np.zeros((grid_x*grid_y,3), np.float32)
        objp[:,:2] = np.mgrid[0:grid_x,0:grid_y].T.reshape(-1, 2)
                
        # process all calibration images
        for filename in glob.glob(path_mask):
            
            print ('... processing image ' + filename, end='')

            img = cv2.imread(filename)
    
            # skip invalid images
            if img is None:
                continue
        
            if img_shape is None:
                img_shape = img.shape[0:2]
    
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            found, corners = cv2.findChessboardCorners(gray, (grid_x, grid_y), None)
    
            if found: 
                objpoints.append(objp)
                imgpoints.append(corners)
                print(' ... OK')
            else:
                print(' ... skipped')
    
        # end for (filenames)
        
        # compute camera-matrix and distance coefficients
        cal_ok, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, img_shape[0:2], None, None)

        if cal_ok:
            self.save()
            return True
        
        return False
    
    def undistort(self, src_img):
        return cv2.undistort(src_img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        
    def save(self):
        with open(self.name + '.json', 'w') as f:
            json.dump({'matrix' : self.camera_matrix.tolist(),
                       'dist' : self.dist_coeffs.tolist()}, f)
    
    def load(self):
        if not os.path.isfile(self.name + '.json'):
            return False
    
        print('Loading calibration data')
        with open(self.name + '.json', 'r') as f:
            data = json.load(f)
            self.camera_matrix = np.array(data['matrix'])
            self.dist_coeffs = np.array(data['dist'])
            return True