#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Sweep Functions
^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "sweep_to_map",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np

from wradlib.georef import misc, projection


def polar_to_cart(coords):
    """Converts polar coordinates to cartesian coordinates
       Uses the radar convention (starting north and going clockwise)

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains polar coordinates (r,a).

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains cartesian coordinates (x, y).
    """
    ranges = coords[..., 0]
    azimuths = coords[..., 1]

    x = ranges * np.cos(np.radians(90 - azimuths))
    y = ranges * np.sin(np.radians(90 - azimuths))

    coords = np.stack((x, y), axis=-1)

    return(coords)


def cart_to_polar(coords):
    """Convert cartesian coordinates to polar coordinates
       Uses the radar convention (starting north and going clockwise)

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains cartesian coordinates (x, y).

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains polar coordinates (r,a).
    """
    x = coords[..., 0]
    y = coords[..., 1]

    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    t = t*180/np.pi
    t = 90 - t
    t[t < 0] = 360 + t[t < 0]

    coords = np.stack((r, t), axis=-1)

    return(coords)


def target_gcz(elangle, ranges, radius=6371E3, ke=4/3):
    """Calculate great circle distance and height of the target point.       
       Based on WMO.

    Parameters
    ----------
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    radius : float
        radius of the spherical model
    ke : float
        Bending model (5/4 can be used for mountain top)

    Returns
    -------
    distance : float
        great circle distance along the sphere
    height : float
        height above radar antenna

    """

    gamma = np.deg2rad(elangle)

    # Effective earth radius
    rk = (re * ke)

    a = ranges
    b = rk
    c = np.sqrt(a**2 + b**2 + 2 * a * b * np.sin(gamma))
    height = c - rk

    distance = rk * np.arcsin(a * np.cos(gamma) / (b + height))

    height = height + height_radar

    return distance, height


def sweep_coordinates(ranges, azimuths):
    """Get the sweep coordinates

    Parameters
    ----------
    ranges : :class:`numpy:numpy.array`
        Contains the radial distances in meters.
    azimuths : :class:`numpy:numpy.array`
        Contains the azimuthal angles in degree.

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrays, nbins, 2) containing the sweep coordinates.

    """

    ranges, azimuths = np.meshgrid(ranges, azimuths)
    sweepcoords = np.stack((ranges, azimuths), axis=-1)

    return(sweepcoords)


def sweep_to_map(sitecoords, elangle, ranges, azimuths,
                 projection=None, altitude=False,
                 binsize=None):
    """Get map coordinates from sweep coordinates

    Parameters
    ----------
    sitecoords : array of floats
        radar site location: longitude, latitude and altitude (amsl)
    elangle: :class:`numpy:numpy.array`
        Contains the elevation angle in degree.
    ranges : :class:`numpy:numpy.array`
        Contains the radial distances in meters.
    azimuths : :class:`numpy:numpy.array`
        Contains the azimuthal angles in degree.
    projection : osr.SpatialReference
        map projection definition
    altitude : bool
        True to get also the altitude
        for the given projection

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrays, nbins, ndim) containing the map coordinates.
        If binsize is not None, an array

    """
    sitecoords = georef.geoid_to_ellipsoid(sitecoords)

    re = georef.get_earth_radius(sitecoords[1])

    ranges, z = georef.bin_gcz(sitecoords[2], elangle, ranges, re)

    ranges, azimuths = np.meshgrid(ranges, azimuths)
    coords = np.stack((ranges, azimuths), axis=-1)
    coords = georef.polar_to_cart(coords)

    if altitude:
        x = coords[..., 0]
        y = coords[..., 1]
        z = np.tile(z, (x.shape[0], 1))
        coords = np.stack((x, y, z), axis=-1)

    # Reproject if needed
    if projection is not None:
        radar = georef.get_radar_projection(sitecoords)
        coords = georef.reproject(coords, projection_source=radar,
                                  projection_target=projection)

    return coords


def sweep_to_polyvert(*args, binsize=None, ravel=False, **kwargs):
    """Get projected polygon representation of sweep bins

    Parameters
    ----------
    args :
        arguments for sweep_to_map
    binsize : tuple of float
        bin size (range, azimuth)
    kwargs :
        keyword arguments for sweep_to_map
    ravel :
        True to ravel

    Returns
    -------
    vertices : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape(nbins, 5, 2).

    """
    ranges = args[2]
    azimuths = args[3]
    if binsize is None:
        rscale = (ranges[1] - ranges[0])/2
        ascale = (azimuths[1] - azimuths[0])/2
    else:
        rscale, ascale = binsize

    rsta = ranges - rscale
    rend = ranges + rscale
    asta = azimuths - ascale
    aend = azimuths + ascale

    args = args[0:2]

    ulc = sweep_to_map(*args, rsta, asta, **kwargs)
    urc = sweep_to_map(*args, rend, asta, **kwargs)
    lrc = sweep_to_map(*args, rend, aend, **kwargs)
    llc = sweep_to_map(*args, rsta, aend, **kwargs)

    vertices = np.stack((ulc, urc, lrc, llc, ulc), axis=-2)

    if ravel:
        vertices = vertices.reshape((-1, 5, 2))

    return(vertices)


def map_to_sweep(coords, sitecoords, elangle, projection=None):
    """Returns sweep coordinates from map coordinates.

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        array of shape (..., ndim) containing map coordinates.
    sitecoords : array of floats
        radar site location: longitude, latitude and altitude (amsl)
    elangle: :class:`numpy:numpy.array`
        elevation angle in degree.
    projection : osr object
        map projection definition

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2) containing the sweep coordinates.
    """

    coords = coords[..., 0:2]
    sitecoords = georef.geoid_to_ellipsoid(sitecoords)
    if projection is not None:
        radar = georef.get_radar_projection(sitecoords)
        coords = georef.reproject(coords,
                                  projection_source=projection,
                                  projection_target=radar)
    # cartesian to polar
    coords = georef.cart_to_polar(coords)
    dists = coords[..., 0]
    azimuths = coords[..., 1]

    re = 6371007.0
    ke = 4/3
    rk = re * ke

    alpha = dists/rk
    gamma = np.pi/2 + np.deg2rad(elangle)
    beta = np.pi - alpha - gamma
    b = rk + sitecoords[2]
    ranges = b * np.sin(alpha)/np.sin(beta)

    coords = np.stack((ranges, azimuths), axis=-1)

    return coords


def raster_to_sweep(rastercoords, projection,
                    sitecoords, elangle, ranges, azimuths,
                    binsize=None, rastervalues=None,
                    method='area', fill='nearest',
                    **kwargs):
    """
    Map raster values to a radar sweep
    taking into account scale differences.

    Parameters
    ----------
    rastercoords : numpy ndarray
        raster coordinates
    projection : osr.SpatialReference
        raster projection definition
    sitecoords : sequence of 3 floats
        Longitude, Latitude and Altitude (in meters above sea level)
    elangle : float
        elevation angle
    ranges : numpy array
        range coordinates
    azimuths : numpy array
        azimuth coordinates
    binsize : (float, float)
        bin size in range and azimuth
    rastervalues : numpy ndarray
        raster values
    method : string
        interpolation method: nearest, linear, spline, binned or area
    fill : string
        second method to fill holes: nearest, linear, spline
    kwargs : keyword arguments
        keyword arguments to interpolation class

    Returns
    -------
    sweepval : numpy ndarray
        sweep values of size (nrays, nbins)
    """

    if method not in ["nearest", "linear", "spline", "binned", "area"]:
        raise ValueError("Invalid method")

    if binsize is not None and method != "area":
        raise ValueError("Only 'area' method supported with bin size")

    if method in ["linear", "nearest", "spline"]:

        fill = None

        sweepcoords = sweep_to_map(sitecoords, elangle, ranges, azimuths,
                                   projection)

        if method == "spline":
            interpolator = ipol.RectSpline(rastercoords, sweepcoords,
                                           **kwargs)
        else:
            interpolator = ipol.RectLinear(rastercoords, sweepcoords,
                                           method, **kwargs)

    if method == "binned":
        rastercoords_sweep = map_to_sweep(rastercoords, sitecoords, elangle,
                                          projection)
        sweepcoords = sweep_coordinates(ranges, azimuths)
        interpolator = ipol.RectBin(rastercoords_sweep, sweepcoords)

    if method == "area":

        edges = util.grid_center_to_edge(rastercoords)
        pixels = util.grid_to_polyvert(edges)

        bins = georef.sweep_to_polyvert(sitecoords, elangle,
                                        ranges, azimuths, binsize,
                                        projection=projection)

        interpolator = ipol.PolyArea(pixels, bins)

    if fill is not None:
        alternative = raster_to_sweep(rastercoords, projection,
                                      sitecoords, elangle, ranges, azimuths,
                                      binsize,
                                      method=fill,
                                      **kwargs)
        interpolator = ipol.Sequence([interpolator, alternative])

    if rastervalues is None:
        return(interpolator)

    sweepval = interpolator(rastervalues)

    return(sweepval)


def raster_to_sweep_multi(rasters, *args, **kwargs):
    """
    Map several rasters to a radar sweep

    Parameters
    ----------
    raster : list of gdal.Dataset
        georeferenced raster images
    args : arguments
        passed to raster_to_sweep
    kwargs : keyword arguments
        passed to raster_to_sweep
    """
    sweepval = None

    for raster in rasters:
        rastervalues = georef.read_gdal_values(raster)
        rastercoords = georef.read_gdal_coordinates(raster)
        projection = georef.read_gdal_projection(raster)
        temp = raster_to_sweep(rastercoords, projection,
                               *args, rastervalues=rastervalues, **kwargs)
        if sweepval is None:
            sweepval = temp
            continue
        bad = np.isnan(temp)
        sweepval[~bad] = temp[~bad]

    return(sweepval)


def sweep_to_raster(rastercoords, projection,
                    sitecoords, elangle, ranges, azimuths,
                    binsize=None, sweepvalues=None,
                    method='area', fill='nearest',
                    **kwargs):
    """
    Map sweep bin values to raster cells

    Parameters
    ----------
    rastercoords : numpy ndarray
        raster coordinates
    projection : osr.SpatialReference
        raster projection definition
    sitecoords : array of floats
        Longitude, Latitude and Altitude (in meters above sea level)
    elangle : float
        elevation angle
    ranges : numpy array
        sweep ranges bin center
    azimuths : numpy array
        sweep azimuth edges
    binsize : float
        bin size in range and azimuth
    sweepvalues : numpy ndarray
        sweep values
    method : string
        interpolation method: nearest, linear, spline, binned or area
    fill : string
        second method to fill holes: nearest, linear, spline
    kwargs : keyword arguments
        keyword arguments to raster_to_sweep_method

    Returns
    -------
    rastervalues : numpy ndarray
        rastervalues values of size (nrows, ncols)
    """

    if method not in ["nearest", "linear", "spline", "binned", "area"]:
        raise ValueError("Invalid method")

    if binsize is not None and method != "area":
        raise ValueError("Only 'area' method supported with bin size")

    if method in ["linear", "nearest", "spline"]:

        fill = None

        myrastercoords = map_to_sweep(rastercoords, sitecoords, elangle,
                                      projection)
        sweepcoords = sweep_coordinates(ranges, azimuths)

        if method == "spline":
            interpolator = ipol.RectSpline(sweepcoords, myrastercoords,
                                           **kwargs)
        else:
            interpolator = ipol.RectLinear(sweepcoords, myrastercoords,
                                           method, **kwargs)

    if method == "binned":

        sweepcoords = sweep_to_map(sitecoords, elangle, ranges, azimuths,
                                   projection)
        interpolator = ipol.RectBin(sweepcoords, rastercoords)

    if method == "area":

        edges = util.grid_center_to_edge(rastercoords)
        pixels = util.grid_to_polyvert(edges)

        bins = georef.sweep_to_polyvert(sitecoords, elangle,
                                        ranges, azimuths, binsize,
                                        projection=projection)

        interpolator = ipol.PolyArea(bins, pixels)

    if fill is not None:
        alternative = sweep_to_raster(rastercoords, projection,
                                      sitecoords, elangle, ranges, azimuths,
                                      binsize,
                                      method=fill,
                                      **kwargs)
        interpolator = ipol.Sequence([interpolator, alternative])

    if sweepvalues is None:
        return(interpolator)

    rastervalues = interpolator(sweepvalues)

    return(rastervalues)

