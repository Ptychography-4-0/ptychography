# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:41:54 2020

@author: oleh.melnyk
"""
import numpy as np
from skimage import data
from ptychography40.stitching.stitching import stitch


def test_stitching():
    img_re = data.camera()
    img_im = data.coins()

    d = min(img_re.shape[0], img_re.shape[1], img_im.shape[0], img_im.shape[1])
    psize = 150
    shift = 75
    # noise_level = 1000

    d = (d//psize)*psize

    obj = img_re[:d, :d] + 1.0j * img_im[:d, :d]

    locations_1d = np.array(range(0, d-psize+1, shift))
    locations_2d = np.zeros((len(locations_1d)**2, 2), dtype=int)
    locations_2d[:, 0] = np.repeat(locations_1d, len(locations_1d))
    locations_2d[:, 1] = np.tile(locations_1d, len(locations_1d))

    R = locations_2d.shape[0]

    parts = np.zeros((d, d, R), dtype=complex)
    for r in range(R):
        loc = locations_2d[r, :]

        #    cut part
        parts[loc[0]:(loc[0] + psize), loc[1]:(loc[1] + psize), r] = obj[
            loc[0]:(loc[0] + psize), loc[1]:(loc[1] + psize)
        ]

        # #    add gaussian noise
        # noise= np.random.normal(size=(psize,psize)) + 1.0j*np.random.normal(size=(psize,psize))
        # noise *= np.sqrt(noise_level)

        # parts[loc[0]:(loc[0] + psize),loc[1]:(loc[1] + psize),r] += noise

        #    add random phase rotation
        angle = np.random.normal(1) + 1.0j*np.random.normal(1)
        angle = angle / abs(angle)
        if r == 0:
            angle = 1
        parts[loc[0]:(loc[0] + psize), loc[1]:(loc[1] + psize), r] *= angle

    known_good = obj
    your_result = stitch(parts)
    assert np.allclose(known_good, your_result)
