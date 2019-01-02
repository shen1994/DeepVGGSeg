# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:40:37 2018

@author: shen1994
"""

import cv2
import numpy as np

class Generator:
    
    def __init__(self,
                 image_path,
                 label_path,
                 trainval_path,
                 image_shape=(224, 224, 3),
                 classes=21,
                 is_enhance=True):
        
        self.image_shape = image_shape
        self.is_enhance = is_enhance
        
        with open('images/ImageSets/Segmentation/trainval.txt', 'r') as f:
            ipaths = f.readlines()
            ipaths = [ipath.replace('\n', '') for ipath in ipaths]

        self.samples = len(ipaths)
        self.classes = classes
        
        self.sample_paths = [image_path + '/' + ipath + '.jpg' for ipath in ipaths]
        self.result_paths = [label_path + '/' + ipath + '.npy' for ipath in ipaths]

        self.dataset_bgr_mean = [103.939, 116.779, 123.680]

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])
        
    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = np.random.random() + 0.5 
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = np.random.random() + 0.5
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = np.random.random() + 0.5
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * 0.5
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def generate(self, batch_size=32):
        
        while(True):
        
            indexes = np.arange(0, self.samples)
            np.random.shuffle(indexes)
            sample_paths = [self.sample_paths[i] for i in indexes]
            result_paths = [self.result_paths[i] for i in indexes]
    
            counter = 0
            x_batch, y_batch = [], []
            for i in range(self.samples):
                image = cv2.imread(sample_paths[i], 1).astype(np.float)                
                if self.is_enhance:
                    image = self.saturation(image)
                    image = self.brightness(image)
                    image = self.contrast(image)
                    image = self.lighting(image)
                image = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))
                image[:, :, 0] -= self.dataset_bgr_mean[0]
                image[:, :, 1] -= self.dataset_bgr_mean[1]
                image[:, :, 2] -= self.dataset_bgr_mean[2]
                
                label = np.load(result_paths[i])
                label = cv2.resize(label, (self.image_shape[0], self.image_shape[1]))
                trans_label = np.zeros((self.image_shape[0], self.image_shape[1], self.classes))
                for c in range(self.classes):
                    trans_label[:, :, c] = (label == c).astype(int)
                trans_label = np.reshape(trans_label, (self.image_shape[0] * self.image_shape[1], self.classes))
                
                counter += 1
                
                x_batch.append(image)
                y_batch.append(trans_label)
                
                if counter == batch_size: 
                    yield np.array(x_batch), np.array(y_batch)
                    
                    counter = 0
                    x_batch, y_batch = [], []
                
if __name__ == "__main__":

    generator = Generator('images/JPEGImages', 'images/SegClass', 
                          'images/ImageSets/Segmentation/trainval.txt')
                
    x, y = generator.generate(batch_size=1).__next__()            
    print(x.shape, y.shape)            
                
                
                
                
                
                