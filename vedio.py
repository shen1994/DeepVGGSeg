# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:30:25 2018

@author: shen1994
"""

import os
import cv2
import numpy as np

from model import VGGUnet

color_map = [(  0,   0,   0), (128,    0,  0), (  0, 128,   0), (128, 128,   0),
             (  0,   0, 128), (128,   0, 128), (  0, 128, 128), (128, 128, 128),
             ( 64,   0,   0), (192,   0,   0), ( 64, 128,   0), (192, 128,   0),
             ( 64,   0, 128), (192,   0, 128), (  6, 128, 128), (192, 128, 128),
             (  0,  64,   0), (128,  64,   0), (  0, 192,   0), (128, 192,   0),
             (  0,  64, 128)]

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    image_shape = (256, 256, 3)
    classes = 21
        
    model = VGGUnet(input_shape=image_shape, n_classes=classes)    
    model.load_weights('model/weights.19-100-0.15.hdf5', by_name=True)
    
    # camera settins
    vedio_shape = [1920, 1080]
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vedio_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vedio_shape[1])
    cv2.namedWindow("Deep3DPose", cv2.WINDOW_NORMAL)

    while(True):
        
        _, o_image = cap.read()
        
        image_width, image_height = o_image.shape[0], o_image.shape[1]
        image = cv2.resize(o_image, (image_shape[0], image_shape[1]))
        image = image / 255.
        
        label = model.predict(np.array([image]))[0]                                 
         
        label = np.argmax(label, axis=1)
        label = label.reshape((image_shape[0]//2,  image_shape[1]//2))
        result = np.zeros((image_shape[0]//2, image_shape[1]//2, 3))
        for c in range(classes):
            result[:,:,0] += (label[:,:] == c) * color_map[c][0]
            result[:,:,1] += (label[:,:] == c) * color_map[c][1]
            result[:,:,2] += (label[:,:] == c) * color_map[c][2]
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (image_width, image_height))
    
        cv2.imshow("Deep3DPose", result)
            
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
       
    cv2.destroyAllWindows()
