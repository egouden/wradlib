#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Raster Functions
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "read_gdal_values",
    "read_gdal_projection",
    "read_gdal_coordinates",
    "extract_raster_dataset",
    "get_raster_extent",
    "get_raster_elevation",
    "reproject_raster_dataset",
    "merge_raster_datasets",
    "create_raster_dataset",
    "set_raster_origin",
    "set_raster_indexing",
    "set_coordinate_indexing",
    "raster_to_polyvert",
    "create_raster_geographic",
    "create_raster_xarray",    
    "raster_values",
    "raster_coordinates",
    "raster_extract",
    "raster_extent",
    "raster_elevation",
]
__doc__ = __doc__.format("\n   ".join(__all__))


import numpy
import rasterio.enums
import affine
import xarray

import wradlib
from wradlib import georef
from wradlib.util import import_optional, warn

gdal = import_optional("osgeo.gdal")
gdal_array = import_optional("osgeo.gdal_array")
osr = import_optional("osgeo.osr")


def _pixel_coordinates(nx, ny, mode):
    """Get the pixel coordinates of an image.

    Parameters
    ----------
    nx : int
        x size (number of columns)
    ny : int
        y size (numbers or rows)
    mode : str
        either 'center' (0.5 1.5 ...) or 'edge' (0 1 ...)

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
         array containing pixel coordinates (x,y) in image convention
         shape is (nrows, ncols, 2) if mode==center
         shape is (nrows+1, ncols+1, 2) if mode==edge

    """
    if mode == "center":
        x = np.linspace(0.5, nx - 0.5, num=nx)
        y = np.linspace(0.5, ny - 0.5, num=ny)

    if mode == "edge":
        x = np.linspace(0, nx, num=nx + 1)
        y = np.linspace(0, ny, num=ny + 1)

    X, Y = np.meshgrid(x, y)
    coordinates = np.stack((X, Y), axis=-1)

    return coordinates


def _pixel_to_map(coordinates, geotransform):
    """Apply a geographical transformation to return map coordinates from
    pixel coordinates.

    Parameters
    ----------
    coordinates : :class:`numpy:numpy.ndarray`
        2d array of pixel coordinates
    geotransform : :class:`numpy:numpy.ndarray`
        geographical transformation vector:

            - geotransform[0] = East/West location of Upper Left corner
            - geotransform[1] = X pixel size
            - geotransform[2] = X pixel rotation
            - geotransform[3] = North/South location of Upper Left corner
            - geotransform[4] = Y pixel rotation
            - geotransform[5] = Y pixel size

    Returns
    -------
    coordinates_map : :class:`numpy:numpy.ndarray`
        2d array with map coordinates (x,y)
    """
    coordinates_map = np.empty(coordinates.shape)
    coordinates_map[..., 0] = (
        geotransform[0]
        + geotransform[1] * coordinates[..., 0]
        + geotransform[2] * coordinates[..., 1]
    )
    coordinates_map[..., 1] = (
        geotransform[3]
        + geotransform[4] * coordinates[..., 0]
        + geotransform[5] * coordinates[..., 1]
    )
    return coordinates_map


def read_gdal_coordinates(dataset, *, mode="center"):
    """Get the projected coordinates from a GDAL dataset.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing
    mode : str
        either 'center' or 'edge'

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
        Array of shape (nrows,ncols,2) containing xy coordinates.
        The array indexing follows image convention with origin
        at upper left pixel.
        The shape is (nrows+1,ncols+1,2) if mode == edge.

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_cloud.ipynb`.

    """
    coordinates_pixel = _pixel_coordinates(
        dataset.RasterXSize, dataset.RasterYSize, mode
    )

    geotransform = dataset.GetGeoTransform()
    coordinates = _pixel_to_map(coordinates_pixel, geotransform)

    return coordinates


def read_gdal_projection(dataset):
    """Get a projection (OSR object) from a GDAL dataset.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        dataset projection object

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_cloud.ipynb`.

    """
    wkt = dataset.GetProjection()
    crs = osr.SpatialReference()
    crs.ImportFromWkt(wkt)
    return crs


def read_gdal_values(dataset, *, nodata=None):
    """Read values from a gdal object.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing
    nodata : float
        replace nodata values

    Returns
    -------
    values : :class:`numpy:numpy.ndarray`
        Array of shape (nrows, ncols) or (nbands, nrows, ncols)
        containing the data values.

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_cloud.ipynb`.

    """
    nbands = dataset.RasterCount

    # data values
    bands = []
    for i in range(nbands):
        band = dataset.GetRasterBand(i + 1)
        nd = band.GetNoDataValue()
        data = band.ReadAsArray()
        if nodata is not None:
            data[data == nd] = nodata
        bands.append(data)

    return np.squeeze(np.array(bands))


def extract_raster_dataset(dataset, *, mode="center", nodata=None):
    """Extract data, coordinates and projection information

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster dataset
    mode : str
        either 'center' or 'edge'
    nodata : float
        replace nodata values

    Returns
    -------
    values : :class:`numpy:numpy.ndarray`
        Array of shape (nrows, ncols) or (nbands, nrows, ncols)
        containing the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrows,ncols,2) containing xy coordinates.
        The array indexing follows image convention with origin
        at the upper left pixel (northup).
        The shape is (nrows+1,ncols+1,2) if mode == edge.
    projection : :py:class:`gdal:osgeo.osr.SpatialReference`
        Spatial reference system of the used coordinates.
    """

    values = read_gdal_values(dataset, nodata=nodata)

    coords = read_gdal_coordinates(dataset, mode=mode)

    projection = read_gdal_projection(dataset)

    return values, coords, projection




def get_raster_extent(dataset, *, geo=False, window=True):
    """Get the coordinates of the 4 corners of the raster dataset

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)
    geo : bool
        True to get geographical coordinates
    window : bool
        True to get the window containing the corners

    Returns
    -------
    extent : :class:`numpy:numpy.ndarray`
        corner coordinates [ul,ll,lr,ur] or
        window extent [xmin, xmax, ymin, ymax]
    """

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    geotrans = dataset.GetGeoTransform()
    xmin = geotrans[0]
    ymax = geotrans[3]
    xmax = geotrans[0] + geotrans[1] * x_size
    ymin = geotrans[3] + geotrans[5] * y_size

    extent = np.array([[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]])

    if geo:
        crs = read_gdal_projection(dataset)
        extent = georef.reproject(extent, src_crs=crs)

    if window:
        x = extent[:, 0]
        y = extent[:, 1]
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

    return extent


def get_raster_elevation(dataset, *, resample=None, **kwargs):
    """Return surface elevation corresponding to raster dataset
       The resampling algorithm is chosen based on scale ratio

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)
    resample : :py:class:`gdal:osgeo.gdalconst.ResampleAlg`
        If None the best algorithm is chosen based on scales.
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12
    kwargs : dict
        keyword arguments passed to :func:`wradlib.io.dem.get_srtm`

    Returns
    -------
    elevation : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing elevation
    """
    extent = get_raster_extent(dataset)
    src_ds = wradlib.io.dem.get_srtm(extent, **kwargs)

    driver = gdal.GetDriverByName("MEM")
    dst_ds = driver.CreateCopy("ds", dataset)

    if resample is None:
        src_gt = src_ds.GetGeoTransform()
        dst_gt = dst_ds.GetGeoTransform()
        src_scale = min(abs(src_gt[1]), abs(src_gt[5]))
        dst_scale = min(abs(dst_gt[1]), abs(dst_gt[5]))
        ratio = dst_scale / src_scale

        resample = gdal.GRA_Bilinear
        if ratio > 2:
            resample = gdal.GRA_Average
        if ratio < 0.5:
            resample = gdal.GRA_NearestNeighbour

    gdal.Warp(
        dst_ds,
        src_ds,
        dstSRS=dst_ds.GetProjection(),
        srcSRS=src_ds.GetProjection(),
        resampleAlg=resample,
    )
    elevation = read_gdal_values(dst_ds)

    return elevation


def set_raster_origin(data, coords, direction):
    """Converts Data and Coordinates Origin

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    direction : str
        'lower' or 'upper', direction in which to convert data and coordinates.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    """
    x_sp, y_sp = coords[1, 1] - coords[0, 0]
    origin = "lower" if y_sp > 0 else "upper"
    same = origin == direction
    if not same:
        data = np.flip(data, axis=-2)
        coords = np.flip(coords, axis=-3)

    return data, coords


def set_raster_indexing(data, coords, *, indexing="xy"):
    """Sets Data and Coordinates Indexing Scheme

    This converts data and coordinate layout from row-major to column major indexing.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (..., M, N) containing the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., M, N, 2) containing xy-coordinates.
    indexing : str
        'xy' or 'ij', indexing scheme in which to convert data and coordinates.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (..., N, M) containing the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., N, M, 2) containing xy-coordinates.
    """
    shape = coords.shape[:-1]

    if shape != data.shape:
        raise ValueError(
            f"coordinate shape {coords.shape} and data shape " f"{data.shape} mismatch."
        )

    coords = set_coordinate_indexing(coords, indexing=indexing)

    # if coordinate shape has changed, we need to transform data too
    if coords.shape[:-1] != shape:
        data_shape = tuple(range(data.ndim - 2)) + (-1, -2)
        data = data.transpose(data_shape)

    return data, coords


def set_coordinate_indexing(coords, *, indexing="xy"):
    """Sets Coordinates Indexing Scheme

    This converts coordinate layout from row-major to column major indexing.

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., M, N, 2) containing xy-coordinates.
    indexing : str
        'xy' or 'ij', indexing scheme in which to convert data and coordinates.

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., N, M, 2) containing xy-coordinates.
    """
    is_grid = hasattr(coords, "shape") and coords.ndim >= 3 and coords.shape[-1] == 2
    if not is_grid:
        raise ValueError(
            f"wrong coordinate shape {coords.shape}, " f"(..., M, N, 2) expected."
        )
    if indexing not in ["xy", "ij"]:
        raise ValueError(f"Unknown indexing value {indexing}. Use either `xy` or `ij`.")

    rowcol = coords[0, 0, 1] == coords[0, 1, 1]
    convert = (rowcol and indexing == "ij") or (not rowcol and indexing == "xy")

    if convert:
        coords_shape = tuple(range(coords.ndim - 3)) + (-2, -3, -1)
        coords = coords.transpose(coords_shape)

    return coords


def reproject_raster_dataset(src_ds, **kwargs):
    """Reproject/Resample given dataset according to keyword arguments

    Parameters
    ----------
    src_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)

    Keyword Arguments
    -----------------
    spacing : float
        float or tuple of two floats
        pixel spacing of destination dataset, same unit as pixel coordinates
    size : int
        tuple of two ints
        X/YRasterSize of destination dataset
    resample : :py:class:`gdal:osgeo.gdalconst.ResampleAlg`
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12
    projection_source : :py:class:`gdal:osgeo.osr.SpatialReference`
        source dataset projection, defaults to None (get projection from src_ds)
    projection_target : :py:class:`gdal:osgeo.osr.SpatialReference`
        destination dataset projection, defaults to None
    align : bool or tuple
        If False, there is no destination grid aligment.
        If True, aligns the destination grid to the next integer multiple of
        destination grid.
        If tuple (upper-left x,y-coordinate), the destination grid is aligned to this point.

    Returns
    -------
    dst_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        reprojected/resampled raster dataset
    """

    # checking kwargs
    spacing = kwargs.pop("spacing", None)
    size = kwargs.pop("size", None)
    resample = kwargs.pop("resample", gdal.GRA_Bilinear)
    src_crs = kwargs.pop("src_crs", None)
    trg_crs = kwargs.pop("trg_crs", None)
    align = kwargs.pop("align", False)

    if spacing is None and size is None:
        raise NameError("Either keyword `spacing` or `size` must be given.")

    if spacing is not None and size is not None:
        warn("Both `spacing` and `size` kwargs given, `size` will be ignored.")

    # Get the GeoTransform vector
    src_geo = src_ds.GetGeoTransform()
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    # get extent
    ulx = src_geo[0]
    uly = src_geo[3]
    lrx = src_geo[0] + src_geo[1] * x_size
    lry = src_geo[3] + src_geo[5] * y_size

    extent = np.array([[[ulx, uly], [lrx, uly]], [[ulx, lry], [lrx, lry]]])

    if trg_crs:
        # try to load projection from source dataset if None is given
        if src_crs is None:
            src_proj = src_ds.GetProjection()
            if not src_proj:
                raise ValueError(
                    "`src_ds` is missing projection information, please use `src_crs`-kwarg "
                    "and provide a fitting GDAL OSR SRS object."
                )
            src_crs = osr.SpatialReference()
            src_crs.ImportFromWkt(src_proj)

        # Transformation
        extent = georef.reproject(extent, src_crs=src_crs, trg_crs=trg_crs)

        # wkt needed
        src_crs = src_crs.ExportToWkt()
        trg_crs = trg_crs.ExportToWkt()

    (ulx, uly, urx, ury, llx, lly, lrx, lry) = tuple(list(extent.flatten().tolist()))

    # align grid to destination raster or UL-corner point
    if align:
        try:
            ulx, uly = align
        except TypeError:
            pass

        ulx = int(max(np.floor(ulx), np.floor(llx)))
        uly = int(min(np.ceil(uly), np.ceil(ury)))
        lrx = int(min(np.ceil(lrx), np.ceil(urx)))
        lry = int(max(np.floor(lry), np.floor(lly)))

    # calculate cols/rows or xspacing/yspacing
    if spacing:
        try:
            x_ps, y_ps = spacing
        except TypeError:
            x_ps = spacing
            y_ps = spacing

        cols = int(abs(lrx - ulx) / x_ps)
        rows = int(abs(uly - lry) / y_ps)
    elif size:
        cols, rows = size
        x_ps = x_size * src_geo[1] / cols
        y_ps = y_size * abs(src_geo[5]) / rows

    # create destination in-memory raster
    mem_drv = gdal.GetDriverByName("MEM")

    # and set RasterSize according ro cols/rows
    dst_ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Float32)

    # Create the destination GeoTransform with changed x/y spacing
    dst_geo = (ulx, x_ps, src_geo[2], uly, src_geo[4], -y_ps)

    # apply GeoTransform to destination dataset
    dst_ds.SetGeoTransform(dst_geo)

    # apply Projection to destination dataset
    if trg_crs is not None:
        dst_ds.SetProjection(trg_crs)
    else:
        dst_ds.SetProjection(src_ds.GetProjection())

    # nodata handling, need to initialize dst_ds with nodata
    src_band = src_ds.GetRasterBand(1)
    nodata = src_band.GetNoDataValue()
    dst_band = dst_ds.GetRasterBand(1)
    if nodata is not None:
        dst_band.SetNoDataValue(nodata)
        dst_band.WriteArray(np.ones((rows, cols)) * nodata)
    dst_band.FlushCache()

    # resample and reproject dataset
    gdal.Warp(
        dst_ds,
        src_ds,
        dstSRS=trg_crs,
        srcSRS=src_crs,
        resampleAlg=resample,
    )
    return dst_ds


def create_raster_dataset(data, coords, *, crs=None, nodata=-9999):
    """Create In-Memory Raster Dataset

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrows, ncols, 2) containing pixel center coordinates
        or
        Array of shape (nrows+1, ncols+1, 2) containing pixel edge coordinates
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Spatial reference system of the used coordinates, defaults to None.
    nodata : int
        Value of NODATA

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        In-Memory raster dataset

    Note
    ----
    The origin of the provided data and coordinates is UPPER LEFT.
    """

    # align data
    data = data.copy()
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, rows, cols = data.shape

    # create In-Memory Raster with correct dtype
    mem_drv = gdal.GetDriverByName("MEM")
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
    dataset = mem_drv.Create("", cols, rows, bands, gdal_type)

    # initialize geotransform
    x_ps, y_ps = coords[1, 1] - coords[0, 0]
    if data.shape[-2:] == coords.shape[0:2]:
        upper_corner_x = coords[0, 0, 0] - x_ps / 2
        upper_corner_y = coords[0, 0, 1] - y_ps / 2
    else:
        upper_corner_x = coords[0, 0, 0]
        upper_corner_y = coords[0, 0, 1]
    geotran = [upper_corner_x, x_ps, 0, upper_corner_y, 0, y_ps]
    dataset.SetGeoTransform(geotran)

    if crs:
        dataset.SetProjection(crs.ExportToWkt())

    # set np.nan to nodata
    dataset.GetRasterBand(1).SetNoDataValue(nodata)

    for i, band in enumerate(data, start=1):
        dataset.GetRasterBand(i).WriteArray(band)

    return dataset


def merge_raster_datasets(datasets, **kwargs):
    """Merge rasters.

    Parameters
    ----------
    datasets : list
        list of :py:class:`gdal:osgeo.gdal.Dataset`
        raster images with georeferencing
    kwargs : dict
        keyword arguments passed to gdal.Warp()

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        merged raster dataset
    """

    dataset = gdal.Warp("", datasets, format="MEM", **kwargs)

    return dataset


def raster_to_polyvert(dataset):
    """Get raster polygonal vertices from gdal dataset.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)

    Returns
    -------
    polyvert : :class:`numpy:numpy.ndarray`
        A N-d array of polygon vertices with shape (..., 5, 2).

    """
    rastercoords = get_coordinates(dataset, mode="edge")

    polyvert = georef.grid_to_polyvert(rastercoords)

    return polyvert


def create_raster_xarray(crs, bounds, res, vars=None):
    """
    Create a georeferenced xarray.Dataset from bounds and resolution,
    with only spatial dimensions defined (no coordinate values),
    optionally populated with 2D data variables.

    Parameters
    ----------
    crs : str or rasterio.crs.CRS
        Coordinate Reference System identifier (e.g., 'EPSG:3857').    
    bounds : tuple of float
        (minx, miny, maxx, maxy) in target CRS units.
    res : float or (float, float)
        Pixel size in CRS units. If a single value is given, square pixels are assumed.
    vars : dict, optional
        Dictionary of variable names to 2D arrays with shape (height, width).

    Returns
    -------
    xr.Dataset
        Dataset with 'y', 'x' dimensions (no coordinate values) and optional
        data variables, georeferenced via .rio.

    Raises
    ------
    ValueError
        If bounds do not match resolution exactly.
        If any variable shape does not match (height, width).
    """
    minx, miny, maxx, maxy = bounds

    if isinstance(res, (int, float)):
        xres = yres = float(res)
    else:
        xres, yres = res

    x_extent = maxx - minx
    y_extent = maxy - miny
    nx = x_extent / xres
    ny = y_extent / yres

    if not (nx.is_integer() and ny.is_integer()):
        raise ValueError(f"Bounds do not match resolution exactly:\n  nx={nx}, ny={ny}")

    width = int(nx)
    height = int(ny)

    transform = affine.Affine(xres, 0, minx, 0, -yres, maxy)

    # Create dataset with empty dimensions only
    ds = xr.Dataset()
    ds = ds.assign_coords({})  # no spatial coords
    ds = ds.assign_dims({"y": height, "x": width})
    ds = ds.rio.write_crs(crs).rio.write_transform(transform)

    # Add variables if provided
    if vars:
        for name, array in vars.items():
            if array.shape != (height, width):
                raise ValueError(
                    f"Variable '{name}' has shape {array.shape}, expected ({height}, {width}) for ('y', 'x') dimensions"
                )
            ds[name] = (("y", "x"), array)

    return ds


def create_raster_geographic(
    bounds: tuple[int, int, int, int],
    resolution: int | tuple[int, int],
    resolution_unit: str = "meters",
    vars: dict[str, np.ndarray] | None = None
) -> "xarray.Dataset":
    """
    Create a geographic raster with exact locations.

    Parameters
    ----------
    bounds : tuple[int, int, int, int]
        Geographic extent in integer arcseconds:
        (min_longitude_arcsec, max_longitude_arcsec,
         min_latitude_arcsec, max_latitude_arcsec).

    resolution : int or tuple[int, int]
        Size of a single grid cell in units given by `resolution_unit`.
        - Single int: applied to both X (longitude) and Y (latitude).
        - Tuple: interpreted as (res_x, res_y).

    resolution_unit : {"meters", "arcseconds"}, default="meters"
        Unit in which `resolution` is specified.
        - "meters": converted to arcseconds and snapped to fit bounds exactly.
        - "arcseconds": used directly.

    vars : dict[str, numpy.ndarray], optional
        Dictionary mapping layer name (string) to a NumPy array containing
        the layer’s data. Arrays must match the raster grid’s shape.

    Returns
    -------
    xarray.Dataset
        Dataset with integer‑arcsecond WGS84 coordinates aligned
        exactly to the specified bounds and resolution.
    """
    if resolution_unit not in ("meters", "arcseconds"):
        raise ValueError(f"Unsupported resolution_unit: {resolution_unit}")

    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    if resolution_unit == "meters":
        lat_mid_deg = (bounds[2] + bounds[3]) / 2 / 3600.0
        res_x_arc = meters_to_arcseconds_lon(resolution[0], lat_mid_deg)
        res_y_arc = meters_to_arcseconds_lat(resolution[1])
        extent_x_arc = bounds[1] - bounds[0]
        extent_y_arc = bounds[3] - bounds[2]
        res_x_arc = snap_to_extent(extent_x_arc, res_x_arc)
        res_y_arc = snap_to_extent(extent_y_arc, res_y_arc)
        resolution = (int(round(res_x_arc)), int(round(res_y_arc)))

    crs = georef.wgs84_arcseconds_crs()

    dataset = create_raster_xarray(
        bounds,
        resolution,
        crs,
        vars
    )
    return dataset


def raster_extract(
    ds: xarray.Dataset,
    *,
    variable: str | None = None,
    mode: str = "center",
    nodata: float | None = None
) -> tuple[numpy.ndarray, numpy.ndarray, str]:
    """
    Extract raster values, coordinates, and projection from a rioxarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Raster dataset loaded via rioxarray.
    variable : str or None, optional
        Name of the data variable to extract. If None, uses the first variable.
    mode : str, optional
        Pixel coordinate mode: 'center' or 'edge'.
    nodata : float or None, optional
        Value to treat as nodata. If specified, replaces matching values with NaN.

    Returns
    -------
    tuple of numpy.ndarray, numpy.ndarray, str
        - values : Array of shape (bands, rows, cols) or (rows, cols) if single band.
        - coordinates : Array of shape (rows, cols, 2) or (rows+1, cols+1, 2) if mode == 'edge'.
        - projection : CRS in WKT format.
    """
    if variable is None:
        variable = list(ds.data_vars)[0]

    values = raster_values(ds, variable, nodata=nodata)
    coordinates = raster_coordinates(ds, variable, mode=mode)
    projection = ds.rio.crs.to_wkt()

    return values, coordinates, projection


def raster_values(
    ds: xarray.Dataset,
    variable: str,
    nodata: float | None = None
) -> numpy.ndarray:
    """Extract raster values from a Dataset variable, replacing nodata if specified."""
    da = ds[variable]
    if nodata is not None:
        da = da.where(da != nodata, numpy.nan)
    return da.values


def raster_coordinates(
    ds: xarray.Dataset,
    variable: str,
    mode: str = "center"
) -> numpy.ndarray:
    """Compute pixel coordinates from a Dataset variable."""
    da = ds[variable]

    if "x" not in da.coords or "y" not in da.coords:
        da = da.rio.reproject(da.rio.crs)

    x = da.x.values
    y = da.y.values

    if mode == "center":
        xx, yy = numpy.meshgrid(x, y)
    elif mode == "edge":
        xx, yy = numpy.meshgrid(_compute_edges(x), _compute_edges(y))
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'center' or 'edge'.")

    return numpy.stack([xx, yy], axis=-1)


def _compute_edges(coords: numpy.ndarray) -> numpy.ndarray:
    """Compute pixel edge coordinates from center coordinates."""
    diffs = numpy.diff(coords) / 2
    edges = numpy.empty(coords.size + 1)
    edges[1:-1] = coords[:-1] + diffs
    edges[0] = coords[0] - diffs[0]
    edges[-1] = coords[-1] + diffs[-1]
    return edges


def raster_extent(
    ds: xarray.Dataset,
    *,
    geo: bool = False
) -> numpy.ndarray:
    """
    Compute the bounding box extent of a rioxarray Dataset,
    correcting for pixel-center coordinates and optionally
    reprojecting to geographic coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Raster dataset loaded via rioxarray.
    geo : bool, optional
        If True, reproject extent to geographic coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape (4,) with extent [xmin, ymin, xmax, ymax].
    """
    x = ds.x.values
    y = ds.y.values

    dx = numpy.abs(x[1] - x[0])
    dy = numpy.abs(y[1] - y[0])

    xmin = x.min() - dx / 2
    xmax = x.max() + dx / 2
    ymin = y.min() - dy / 2
    ymax = y.max() + dy / 2

    if geo:
        xmin, ymin, xmax, ymax = ds.rio.transform_bounds(
            dst_crs="EPSG:4326",
            left=xmin,
            bottom=ymin,
            right=xmax,
            top=ymax
        )

    extent = numpy.array([xmin, ymin, xmax, ymax], dtype=numpy.float64)
    return extent


def choose_rioxarray_resampling(ratio: float) -> rasterio.enums.Resampling:
    if ratio > 2:
        return rasterio.enums.Resampling.average
    elif ratio < 0.5:
        return rasterio.enums.Resampling.nearest
    else:
        return rasterio.enums.Resampling.bilinear


def raster_elevation(
    ds: xarray.Dataset,
    **kwargs
) -> xarray.Dataset:
    extent = get_extent(ds, geo=True)
    dem = wradlib.io.dem.get_srtm(extent, **kwargs)

    dem_res = abs(dem.rio.resolution()[0])
    ds_res = abs(ds.rio.resolution()[0])
    ratio = ds_res / dem_res

    resampling_method = choose_rioxarray_resampling(ratio)
    dem = dem.rio.reproject_match(ds, resampling=resampling_method)

    ds = ds.assign_coords(elevation=(("y", "x"), dem.values))
    return ds