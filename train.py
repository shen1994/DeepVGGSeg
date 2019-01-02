# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:33:16 2018

@author: shen1994
"""

import os
from keras.optimizers import Adam
from model import VGGUnet
from generate import Generator
from show import show_predictor

# def facal_loss(y_true, y_pred, alpha=0.2, gamma=0.25, classes=21):
    
#     for i in range(classes)

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
   
    epochs = 1000
    batch_size = 16
    image_shape = (256, 256, 3)
    classes = 21
        
    model = VGGUnet(input_shape=image_shape, n_classes=classes)
    model.load_weights('model/weights.606-00-0.06.hdf5', by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    generator = Generator('images/JPEGImages', 'images/SegClass', 
                          'images/ImageSets/Segmentation/trainval.txt',
                          image_shape=image_shape, classes=classes)
    
    for epoch in range(epochs):
        steps = generator.samples // batch_size
        for step in range(steps):
            X, Y = generator.generate(batch_size=batch_size).__next__()
            # at the begining, the accurary will be 75%, but it's not the truth
            # because the the area of the background is vary large
            # what we need to prove is adding the weights of object
            coss, acc = model.train_on_batch(X, Y)
            
            if step % 100 == 0:
                # 1. show coss
                print("steps: %d, step: %d, coss: %.2f, acc: %.2f" %(steps, step, coss, acc))
                
                # 2. save models
                model.save_weights("model/weights.%02d-%02d-%.2f.hdf5" %(epoch, step, coss))
                
                # 3. show test images 
                show_predictor(X, classes, model, batch_size//2, 'Deep2DSeg')
    