# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:41:54 2020

@author: oleh.melnyk
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
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


@pytest.mark.parametrize(
        'angle1', (0, 13, 179, 180, 181, 355, 360)
)
@pytest.mark.parametrize(
        'angle2', (0, 13, 179, 180, 181, 355, 360)
)
def test_stitching_reference(angle1, angle2):
    '''
    Confirm that the layer with index 0 is the "anchor" of the fit and
    only subsequent layers are rotated to match it.
    '''
    def distance(angle1, angle2):
        d1 = angle2 - angle1
        d2 = angle2 - angle1 + 360
        result = d2.copy()
        select_first = np.abs(d1) < np.abs(d2)
        result[select_first] = d1[select_first]
        return result

    # wrap-around like np.angle
    if angle1 >= 180:
        angle1 -= 180
    if angle2 >= 180:
        angle2 -= 180
    a = np.zeros((1, 2, 2), dtype=np.complex64)
    a[0, 0, 0] = np.exp(1j*np.pi*(angle1-1)/180)
    a[0, 1, 0] = 0.5 * np.exp(1j*np.pi*(angle1+1)/180)
    a[0, 0, 1] = np.exp(1j*np.pi*(angle2-1)/180)
    a[0, 1, 1] = 0.5 * np.exp(1j*np.pi*(angle2+1)/180)
    stitched_angle = np.angle(stitch(a), deg=True)
    print(angle1, angle2, stitched_angle, stitched_angle - angle1)
    assert_allclose(distance(angle1, stitched_angle), ((-1, 1), ), atol=1e-6, rtol=1e-6)
