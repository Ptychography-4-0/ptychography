# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:30:50 2020

@author: oleh.melnyk
"""

import numpy as np

from scipy.sparse.linalg import eigsh


def stitch(objects, wtype='weighted', threshold=1e-12):
    '''
    In paralelized ptychography, each node returns recovered
    part of the object up to a phase factor. Our goal is to stich those pieces by synchronizing
    phase factors. To do so, we solve the angular synchronization, a
    recovery of the unknown phase factors from their (noisy) pairwise differences
    To obtain these pairwise differences, we compute the inner product between
    two overlapping parts. Since on the overlap two objects are approximately the
    same, it gives us the phase difference. Mathematically, it is given by
    < e^(i a_r1) O_r1, e^(i a_r2) O_r2 > approx. = e^(i [a_r1 - a_r2]) norm(O_r1,r_2)^2,
    where O_r is recovered r-th part of the object,
    O_r1,r2 - part of the object corresponging to the overlap of two parts,
    a_r - unknown phase factor.

    This inner products are stored in the ph_diff matrix. Then, the matrix is entrywise
    normalized to separate phases e^(i [a_r1 - a_r2]) and non-negative norm(O_r1,r_2)^2.

    In angular synchronization we take phase differences as data and
    can use norms as weights to solve (un)weighted least squares problem.
    Problem itself is NP-hard and, thus, its relaxation instead is solved, which is
    simply a computation of the eigenvector corresponding to the smallest eigenvalue
    of the graph laplacian matrix.

    Obtained eigenvector (v in code) is entrywise normalized to get phases.
    Each part of the object is then phase-rotated correspondingly. Finally, we
    join the pieces, avaraging on the overlaps.

    Note: obtained solution is the original object up to a globabl phase factor.

    For more details see Section 2 of https://arxiv.org/pdf/2005.02032.pdf

    Possible speed up of the routine: avoid dot product calculation of between all
    parts, but only those which overlap.

    Parameters
    ----------
    objects : numpy.ndarray
        d1 X d2 X R array containing recovered pieces of object,
        required to be stitched
    wtype : str, optional
        Selected type of weights to be used in the angular synchronization
        There are 2 possible choices:
        'unweighted' or 'weighted', depending on usage or not of the weights for
        angular synchronization (explained above)
    threshold : float, optional
        Minimum absolute value to be considered in finding the phase match for stitching.
'''
    d1 = objects.shape[0]
    d2 = objects.shape[1]
    R = objects.shape[2]

    ph_diff = np.zeros((R, R), dtype=complex)

    for r1 in range(R):
        for r2 in range(r1+1, R):
            ph_diff[r1, r2] = np.einsum('ij,ij', objects[:, :, r1], objects[:, :, r2].conj())
            ph_diff[r2, r1] = ph_diff[r1, r2].conj()

    if wtype == 'weighted':
        weights = np.abs(ph_diff)
    elif wtype == 'unweighted':
        weights = (np.abs(ph_diff) > threshold).astype(np.float)

    idx = weights > 0
    ph_diff[idx] = ph_diff[idx]/np.abs(ph_diff[idx])
    degree = np.sum(weights, axis=1)
    laplacian = np.diag(degree) - ph_diff * weights
    sig, v = eigsh(laplacian, 1, which='SM')
    idx = np.abs(v) > threshold
    phases = v
    phases[idx] = phases[idx]/np.abs(phases[idx])
    phases[~idx] = 1

    v *= v[0].conj()

    result = np.zeros((d1, d2), dtype=complex)
    count = np.zeros((d1, d2))
    for r in range(R):
        result[:, :] += phases[r].conj() * objects[:, :, r]
        count[:, :] += np.abs(objects[:, :, r]) > threshold

    result /= count
    return result
