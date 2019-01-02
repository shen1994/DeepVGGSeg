# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:39:44 2018

@author: shen1994
"""

import os
import cv2
import numpy as np

# (224, 224, 192) --- > 'edge'
color_map = [(  0,   0,   0), (128,    0,  0), (  0, 128,   0), (128, 128,   0),
             (  0,   0, 128), (128,   0, 128), (  0, 128, 128), (128, 128, 128),
             ( 64,   0,   0), (192,   0,   0), ( 64, 128,   0), (192, 128,   0),
             ( 64,   0, 128), (192,   0, 128), ( 6,  128, 128), (192, 128, 128),
             (  0,  64,   0), (128,  64,   0), (  0, 192,   0), (128, 192,   0),
             (  0,  64, 128)]
             
color_dict = dict(zip(color_map, np.arange(0, len(color_map))))
          
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining-table',
              'dog', 'horse', 'motorbike', 'person', 'potted-plant', 'sheep', 
              'sofa', 'train', 'tv-monitor']

if __name__ == "__main__":
    
    if not os.path.exists('images/SegClass'):
        os.mkdir('images/SegClass')
        
    with open('images/ImageSets/Segmentation/trainval.txt', 'r') as f:
        images_path = f.readlines()
        images_path = [image_path.replace('\n', '') for image_path in images_path]
  
    counter = 0
    samples = len(images_path)
    for name in images_path:
        
        image_path = 'images/SegmentationClass/' + name + '.png'
        oimage = cv2.imread(image_path, 1)
        image = cv2.cvtColor(oimage, cv2.COLOR_BGR2RGB)
        
        image_zeros = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                elem = tuple(image[i, j, :])
                if elem in color_map:
                    image_zeros[i, j] = color_dict[elem]

        # np.save('images/SegClass/' + name + '.npy', image_zeros)
        counter += 1
        print(samples, counter, 'images/SegClass/' + name + '.npy')
        '''
        label = cv2.resize(image_zeros, (112, 112))
        trans_label = np.zeros((112, 112, 21))
        for c in range(21):
            trans_label[:, :, c] = (label == c).astype(int)

        trans_label = np.reshape(trans_label, (224 * 224 // 4, 21))
        
        trans_label = np.argmax(trans_label, axis=1)
        trans_label = trans_label.reshape((112, 112))
        result = np.zeros((112, 112, 3))
        for c in range(21):
            result[:,:,0] += (trans_label[:,:] == c)*(color_map[c][0])
            result[:,:,1] += (trans_label[:,:] == c)*(color_map[c][1])
            result[:,:,2] += (trans_label[:,:] == c)*(color_map[c][2])
            
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (224, 224))
        
        while(True):
            cv2.imshow("Deep2DOri", oimage)
            cv2.imshow("Deep2DSeg", result)
            if cv2.waitKey(1) == ord('q'):
                break
        '''
        