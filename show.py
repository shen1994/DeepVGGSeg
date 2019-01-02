# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:14:24 2018

@author: shen1994
"""

import cv2
import numpy as np
from skimage.transform import resize

def get_transpose_axes(n):
    
    if n % 2 == 0:
        y_axes = list(range( 1, n-1, 2 ))
        x_axes = list(range( 0, n-1, 2 ))
    else:
        y_axes = list(range( 0, n-1, 2 ))
        x_axes = list(range( 1, n-1, 2 ))
        
    return y_axes, x_axes, [n-1]

def stack_images(images):
    
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len( images_shape ))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]

    return np.transpose(images, axes=np.concatenate(new_axes)).reshape( new_shape)
    
def resize_images(images, size):

    new_images = []
    for image in images:
        nimage = resize(image, (size, size), preserve_range = True)
        new_images.append(nimage)
    return np.array(new_images)

def show_G(images_A, images_B, batch_size, name):
    
    images_A = resize_images(images_A, 128)
    images_B = resize_images(images_B, 128)
    figure = np.stack([images_A, images_B], axis=1)
    figure = figure.reshape((4, batch_size//4) + figure.shape[1:])
    figure = stack_images(figure).astype(np.uint8)

    cv2.imshow(name, figure) 
    cv2.waitKey(1)
    
color_map = [(  0,   0,   0), (128,    0,  0), (  0, 128,   0), (128, 128,   0),
             (  0,   0, 128), (128,   0, 128), (  0, 128, 128), (128, 128, 128),
             ( 64,   0,   0), (192,   0,   0), ( 64, 128,   0), (192, 128,   0),
             ( 64,   0, 128), (192,   0, 128), (  6, 128, 128), (192, 128, 128),
             (  0,  64,   0), (128,  64,   0), (  0, 192,   0), (128, 192,   0),
             (  0,  64, 128)]
    
def show_predictor(images, classes, model, batch_size, name):
    
    images_A, images_B = [], []
    for index in range(batch_size): 
        
        image = images[index]
        image_width, image_height = image.shape[0], image.shape[1]
        
        label = model.predict(np.array([image]))[0] 
                                       
        label = np.argmax(label, axis=1)
        label = label.reshape((image_width,  image_height))
        result = np.zeros((image_width, image_height, 3))
        for c in range(classes):
            result[:,:,0] += (label[:,:] == c)*(color_map[c][0])
            result[:,:,1] += (label[:,:] == c)*(color_map[c][1])
            result[:,:,2] += (label[:,:] == c)*(color_map[c][2])
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)    
        result = cv2.resize(result, (image_height, image_width))
        
        dataset_bgr_mean = [103.939, 116.779, 123.680]
        image[:, :, 0] += dataset_bgr_mean[0]
        image[:, :, 1] += dataset_bgr_mean[1]
        image[:, :, 2] += dataset_bgr_mean[2]
        images_A.append(image.astype(np.uint8))
        images_B.append(result)
        
    show_G(images_A, images_B, batch_size, name)
                