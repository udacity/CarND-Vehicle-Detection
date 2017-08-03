#! /usr/bin/env python
'''
Run a YOLO_v2 style detection model on test video.
Author: Tawn Kramer
Date: 08/03/2017

Brief: Use YOLO_v2 Neural Network detector to identigy objects. Uses the yad2k
https://github.com/allanzelener/YAD2K.git
This keeps list of bounding boxes identified by class. This will attempt to update
the bounding boxes of cars with lane information and estimated speed. It will use
the average color and position to help match against boxes each frame. It will
apply some smoothing to stabilize the boxes and speed estimates.

'''
import argparse
import colorsys
import imghdr
import os
import random
import math

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import cv2

from yad2k.models.keras_yolo import yolo_eval, yolo_head
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test video..')

parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-i',
    '--input_path',
    help='path input video')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output video')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)
parser.add_argument(
    '--start',
    type=int,
    help='start frame to process, default 0',
    default=0)
parser.add_argument(
    '--end',
    type=int,
    help='end frame to process, default last(-1)',
    default=-1)
parser.add_argument(
    '--only_class',
    help='limit class detection to just one type, named by this string',
    default='None')


def make_mask(img_size, #width, height tuple
    '''
    mask a polygon of the image based on percentages of dimensions
    '''
              horizon_perc, # the upper threshold, as a percent of height
              bottom_perc, #the lower thresh, as a percent of height
              mask_bottom_perc = 1.0, #the lower percent of width
              mask_top_perc = 0.5): #the upper percent of width
    img_width = img_size[0]
    img_height = img_size[1]
    
    centerX = img_width / 2
    
    horizon_y = math.floor(horizon_perc * img_height)
    bottom_y_margin = math.floor(bottom_perc * img_height)
    bottom = img_height - bottom_y_margin
    top = horizon_y
    
    mask_bottom_left_x   = math.floor(centerX - img_width * (mask_bottom_perc * 0.5))
    mask_bottom_right_x  = math.floor(centerX + img_width * (mask_bottom_perc * 0.5))
    mask_top_left_x      = math.floor(centerX - img_width * (mask_top_perc * 0.5))
    mask_top_right_x     = math.floor(centerX + img_width * (mask_top_perc * 0.5))

    mask_points = [(mask_bottom_left_x,  bottom),
                   (mask_top_left_x,     top),
                   (mask_top_right_x,    top), 
                   (mask_bottom_right_x, bottom)]
    
    return mask_points


def perspective_reverse(img, corners_src, corners_dest, img_size):
    '''
    take and image and four src points in a rhombus along the lane lines
    dest points in a more linear quad, warp image to straighten effects
    of perspective transformation.
    '''
    
    src = np.float32(corners_src)
    
    dst = np.float32(corners_dest)
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    invM = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, invM

def make_persp_mat(img):
    '''
    Use the settings from a previous project to create a reverse perspective
    matrix. The warped size is arbitrary. The mask dimensions we arrived at
    through trial and error.
    '''

    img_size = (img.shape[1], img.shape[0])
    warped_size = (1200, 1200)
    
    src_cn = make_mask(img_size, 0.65, 0.05, 0.60, 0.1)
    dest_cn = make_mask(warped_size, 0.1, 0.0, 0.4, 0.36)
    
    warped, M, invM = perspective_reverse(img, src_cn, dest_cn, warped_size)
    return M

def tm(pt_xy, M):
    '''
    perform perspective transform on a single point, given x, y pixel
    and persp matrix M
    return the x, y pixel pair in transformed space
    '''
    pt = np.array([pt_xy])
    pt = np.array([pt])
    res = cv2.perspectiveTransform(pt, M)
    return res[0][0]

def estimate_lane(pt):
    '''
    ad-hoc empiracle estimation fn. This is based on the warped_size 
    in the make_persp_mat function above.
    '''
    x = pt[0]
    print(x)
    if x > 4000:
        return 4
    if x > 2200:
        return 3
    if x > 500:
        return 2
    return 1

class BBox(object):
    '''
    class to represent the tracking state for an object, typically a car
    '''
    def __init__(self, box, class_name, conf_score, color, image):
        self.box = list(box)
        self.class_name = class_name
        self.conf_score = conf_score
        self.color = color
        self.age = 1.0
        self.vel = [0.0, 0.0]
        self.speed = 0.0
        self.tm_center = [0.0, 0.0]
        self.avg_color = self.get_avg_color(image)
        self.obscurred = False
        self.lane = -1

    def update_vel(self):
        #apply current vel
        self.box[0] += self.vel[1]
        self.box[2] += self.vel[1]
        self.box[1] += self.vel[0]
        self.box[3] += self.vel[0]

    def interp(self, box, alpha):
        center = self.get_center_pt()

        self.update_vel()

        n = len(box.box)

        for i in range(n):
            self.box[i] = self.box[i] * (1.0 - alpha) + box.box[i] * alpha

        #update vel based on new center 
        n_center = self.get_center_pt()       
        self.vel = [n_center[0] - center[0], n_center[1] - center[1]]

        self.age = 1.0
        self.obscurred = False

    def get_avg_color(self, image):
        img_np = np.asarray(image)
        top, left, bottom, right = self.box
        rect = img_np[int(top): int(bottom), int(left): int(right), :]
        return rect.mean()

    def update_avg_color(self, image):
        self.avg_color = self.get_avg_color(image)

    def matches_color(self, col, thresh):
        return math.fabs(self.avg_color - col) < thresh

    def pt_inside_box(self, pt):
        top, left, bottom, right = self.box
        x, y = pt
        return x >= left and x <= right and y <= bottom and y >= top

    def show(self):
        top, left, bottom, right = self.box
        x, y = pt        
        print('center %.2f, %.2f' % (x, y))
        print('box %.2f, %.2f, %.2f, %.2f' % (top, left, bottom, right))

    def get_center_pt(self):
        top, left, bottom, right = self.box
        return (left + (right - left) / 2.0, top + (bottom - top) / 2.0)

    def draw(self, image, font, thickness, perspMat):

        self.update_avg_color(image)
        
        pt = self.get_center_pt()
        res = tm(pt, perspMat)
        lane = estimate_lane(res)

        prev_center_y = self.tm_center[1]
        self.tm_center = res
        vel_est_y = res[1] - prev_center_y
        
        if self.obscurred :
            label = 'obscurred'
            self.update_vel()
        elif pt[0] < image.width / 2.0:
            label = 'oncoming'
        elif self.class_name == 'car':
            base = 65.0 #how fast am I going?
            tfactor = 0.05 #how far is one pixel?
            factor = 4.0 #how far is one pixel?
            self.lane = lane

            #feel like this esitmate *should* work better in transformed pixel space
            #speed = base + (vel_est_y * tfactor)

            #I like the values this makes, but really quite a bogus estimate
            speed = base + ((-1 * self.vel[0] + -1 * self.vel[1]) * factor)

            #avg speed with prev speed
            if self.speed == 0.0:
                self.speed = speed
            else:
                self.speed = (self.speed * 0.95 + speed * 0.05)

            label = 'lane({}) {} mph'.format(lane, math.floor(self.speed))
        else:        
            label = '{} {:.2f}'.format(self.class_name, self.conf_score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = self.box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.color)
                
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=self.color)
            
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        
        del draw

    def maybe_obscured(self, others):
        '''
        does my center point overlap with other boxes
        '''
        pt = self.get_center_pt()
        for box in others:
            if box == self:
                continue
            if box.pt_inside_box(pt):
                return True
        return False


class BBoxMan(object):
    '''
    Manage a list of BBox bounding boxes. Attempt to help maintain
    continuity between frames by matching new boxes with old.
    '''
    def __init__(self, alpha = 0.1, conf_thres_change = 0.3):
        self.boxes = []
        self.alpha = alpha
        self.conf_thres_change = conf_thres_change

    def add_box(self, bbox):
        '''
        add a bounding box if it doesn't exist
        if it does, interp with old box
        '''
        pt = bbox.get_center_pt()
        found = False
        candidates = []
        color_thresh = 40.0
        for box in self.boxes:
            if box.pt_inside_box(pt) and box.matches_color(bbox.avg_color, color_thresh):
                candidates.append(box)

        best_match = None
        score = 1000000.0
        
        for box in candidates:
            x = box.get_center_pt()[0]
            dist = math.fabs(x - pt[0])
            if dist < score:
                best_match = box
                score = dist 

        if best_match is None:
            self.boxes.append(bbox)
        else:
            best_match.interp(bbox, self.alpha)
            if bbox.conf_score > best_match.conf_score:
                best_match.conf_score = bbox.conf_score
            elif best_match.conf_score < 0.99:
                best_match.conf_score += 0.01


    def purge(self, decay = 0.1):
        '''
        age boxes we haven't seen in a while and remove them
        '''
        remove_arr = []
        for box in self.boxes:
            box.age -= decay
            if box.age < 0.0:
                '''
                this was my ides to try to keep continuity
                when a box passes behind another. Not working that great.
                if box.maybe_obscured(self.boxes):
                    box.age = 3.0
                    box.obscurred = True
                    box.vel[1] = 0.0 #don't go up
                    continue            
                '''
                remove_arr.append(box)

        for box in remove_arr:
            self.boxes.remove(box)

    def draw(self, image, font, thickness, perspMat):
        for box in self.boxes:
            box.draw(image, font, thickness, perspMat)

def _main(args):

    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    input_path = os.path.expanduser(args.input_path)
    output_path = os.path.expanduser(args.output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)

    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    only_class = None

    bbMan = BBoxMan()

    bbMan.perspMat = None

    if args.only_class != "None":
        only_class = args.only_class

    def process_video_frame(image_np):
        '''
        process video within the scope of main to inherit all the variables
        '''
        
        if bbMan.perspMat is None:
            bbMan.perspMat = make_persp_mat(image_np)
            print('bbMan.perspMat')
            print(bbMan.perspMat)

        image = Image.fromarray(image_np)

        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        print('Found {} boxes'.format(len(out_boxes)))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            #when we have specified just one class to detect, skip it when
            #it does not match
            if(only_class is not None and predicted_class != only_class):
                print('ignoring', predicted_class)
                continue

            col = colors[random.randrange(0, len(colors))]
            bbox = BBox(box, predicted_class, score, col, image)
            bbMan.add_box(bbox)
        
        bbMan.purge()
        bbMan.draw(image, font, thickness, bbMan.perspMat)

        return np.asarray(image)

    '''
    Now setup video clip parser
    '''
    start = args.start
    end = args.end
    clip1 = VideoFileClip(input_path)
    clip1 = clip1.subclip(start, end)
    out_clip = clip1.fl_image(process_video_frame)
    out_clip.write_videofile(output_path, audio=False)
    sess.close()

if __name__ == '__main__':
    _main(parser.parse_args())



