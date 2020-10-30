# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:50:51 2020

@author: oleh.melnyk
"""

import numpy as np
from skimage import data
import matplotlib.pyplot as plt
#import sys
#sys.path.insert(1, '../src/ptychography/')
import ptychography.stitching.stitching as stitch

def show_object(obj):
    
    if (len(obj.shape) < 3):
        obj = np.repeat(obj[:, :,np.newaxis], 1, axis=2)
    
    L = obj.shape[2]
    
#    fig, ax = plt.subplots(nrows = L, ncols = 2, squeeze = False)
#    
#    for l in range(L):
#        img = obj[:,:,l]
#        
#        ax[l,0].imshow(np.real(img))
#        ax[l,0].axis('off')
#        
#        ax[l,1].imshow(np.imag(img))
#        ax[l,1].axis('off')
#        
#    plt.show()
    for l in range(L):
        img = obj[:,:,l]
        
        fig, ax = plt.subplots(nrows = 1, ncols = 2)
        
        ax[0].imshow(np.round(np.real(img)).astype(np.uint8))
        ax[0].axis('off')
        
        ax[1].imshow(np.round(np.imag(img)).astype(np.uint8))
        ax[1].axis('off')
        plt.show()

img_re = data.camera()
img_im = data.coins()

d = min( img_re.shape[0],img_re.shape[1],img_im.shape[0],img_im.shape[1])
psize = 150
shift = 75
noise_level = 1000

d = (d//psize)*psize

obj = img_re[:d,:d] + 1.0j * img_im[:d,:d]
#obj = np.ones((d,d),dtype = complex)

plt.gray()

print('Original object')
show_object(obj)

print('Preparing parts')

locations_1d = np.array(range(0,d-psize+1, shift))
locations_2d = np.zeros((len(locations_1d)**2,2),dtype=int)
locations_2d[:,0] = np.repeat(locations_1d,len(locations_1d))
locations_2d[:,1] = np.tile(locations_1d,len(locations_1d))

R = locations_2d.shape[0]

parts = np.zeros( (d,d,R), dtype= complex)
for r in range(R):
    loc = locations_2d[r,:]
    
#    cut part
    parts[loc[0]:(loc[0] + psize),loc[1]:(loc[1] + psize),r] = obj[loc[0]:(loc[0] + psize),loc[1]:(loc[1] + psize)]
    
#    add gaussian noise
    # noise= np.random.normal(size=(psize,psize)) + 1.0j*np.random.normal(size=(psize,psize))
    # noise *= np.sqrt(noise_level)
    
    # parts[loc[0]:(loc[0] + psize),loc[1]:(loc[1] + psize),r] += noise
    
#    add random phase rotation
    angle = np.random.normal(1) + 1.0j*np.random.normal(1)
    angle = angle/ abs(angle)
    if r == 0: 
        angle = 1
    parts[loc[0]:(loc[0] + psize),loc[1]:(loc[1] + psize),r] *= angle
    
print('Resulting parts of the object')
show_object(parts)

print('Stitching...')
res = stitch.stitch(parts)

print('Recovered stitched image')
show_object(res)