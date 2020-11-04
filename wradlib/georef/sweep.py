#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Sweep Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "maximum_intensity_projection",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np

from wradlib.georef import misc, projection


def bin_gcz(height_radar, elangle, ranges, re=None, ke=4/3):
    """Calculate great circle distance and height of a radar bin.

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth

    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    if re is None:
        re = 6371007.0

    gamma = np.deg2rad(90 + elangle)

    rk = re*ke

    a = ranges
    b = rk + height_radar
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(gamma))
    height = c - rk

    alpha = np.arcsin(a * np.sin(gamma) / c)
    distance = rk * alpha

    return distance, height


def bin_gcz_doviak(height_radar, elangle, ranges, re=6371007.0, ke=4/3):
    """Calculate on ground distance and height of target point (Doviak, Zrnic).

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth

    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    gamma = np.deg2rad(elangle)

    # four third radius model for refraction
    rk = (re * ke)

    a = ranges
    b = rk
    c = np.sqrt(a**2 + b**2 + 2 * a * b * np.sin(gamma))
    height = c - rk

    distance = rk * np.arcsin(a * np.cos(gamma) / (b + height))

    height = height + height_radar

    return distance, height


def bin_gcz_other(height_radar, elangle, ranges, re=6371007.0, ke=4/3):
    """Calculate on ground distance and height of target point.

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth


    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    gamma = np.deg2rad(elangle)

    rk = re * ke

    a = ranges
    b = rk + height_radar
    c = np.sqrt(a**2 + b**2 + 2 * a * b * np.sin(gamma))
    height = c - rk

    tmp1 = a * np.cos(gamma)
    tmp2 = a * np.sin(gamma) + b
    distance = rk * np.arctan(tmp1 / tmp2)

    return distance, height


def bin_gcz_full(height_radar, elangle, ranges, re=6371007.0, ke=4/3):
    """Calculate on ground distance and height of target point.

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth

    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    elangle = np.deg2rad(elangle)

    # four third radius model for refraction
    rk = (re * ke)

    # radius of radar beam curvature
    rc = 1 / (np.cos(elangle) * (ke-1) / (rk))

    # euclidian distance from radar to bin
    alpha = ranges / rc
    distance = 2 * rc * np.sin(alpha/2)

    # height from sphere (cosine rule)
    gamma = np.pi/2 - alpha/2 + elangle
    a = height_radar + re
    b = distance
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    rbin = c
    height = rbin - re

    # arc distance over sphere (sin rule)
    beta = np.arcsin(b * np.sin(gamma) / c)
    arc_distance = re * beta

    return arc_distance, height
