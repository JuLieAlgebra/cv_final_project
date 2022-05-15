from contextlib import contextmanager
import zipfile
import os

import omegaconf
import luigi
import bz2
import pandas as pd
import astropy
import astropy.io.fits
import astropy.wcs.utils
import astropy.coordinates
import numpy as np

from final_project.data_downloader import ImageDownloader

# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)


def get_data_cube(observation: pd.Series) -> np.ndarray:
    """Takes in a row of the tabular data corresponding to one observation and returns (5, dim, dim) datacube of observations"""
    dim = int(omegaconf.OmegaConf.load(os.path.join(abs_path, "conf", "preprocessing.yaml"))["dim"])

    bands = ["u", "g", "r", "i", "z"]
    files = []
    for b in bands:
        files.append(
            f"frame-{b}-{str(observation.run).zfill(6)}-{observation.camcol}-{str(observation.field).zfill(4)}.fits.bz2"
        )

    data_cube = np.zeros((5, dim, dim), dtype=np.float64)
    for i, f in enumerate(files):
        data_cube[i] = get_cropped(observation, f, dim)

    # datacube should be (5, dim, dim)
    return data_cube


def get_cropped(observation: pd.Series, fits_file: str, dim: int) -> np.array:
    """
    Thanks to https://lukaswenzl.com/wold-coordinate-system-astropy/ for the reminders!

    Given a file name, returns the cropped image as np array centered on the observation
    Unzips the file in memory, converts the RA & DEC of obj to pixel coords, centers on object & crops to 32x32.

    :param observation: row in tabular data corresponding to observation
    :param fits: path to corresponding unzipped fits image
    :param dim: int for one side of square image to return (default is 32)
    :return: np.array of one cropped image to load into datacube
    """
    # config path loading
    local_paths = omegaconf.OmegaConf.load(os.path.join(abs_path, "conf", "local_paths.yaml"))
    fits_file_path = os.path.join(local_paths["data"], local_paths["images"], fits_file)

    # reading the compressed file without extracting it
    with bz2.BZ2File(fits_file_path, "rb") as fits:
        # opening the image from the uncompressed read
        with astropy.io.fits.open(fits) as hdulist:

            # header contains meta data about the image
            header = hdulist[0].header
            # create world coordinate system from header
            wcs = astropy.wcs.WCS(header)
            # get a coordinate object of observation
            coord = astropy.coordinates.SkyCoord(ra=observation.ra, dec=observation.dec, unit="deg")
            # pixel coordinates of center of observation
            target = astropy.wcs.utils.skycoord_to_pixel(coord, wcs)
            target = (int(target[0]), int(target[1]))
            # actual image
            data = hdulist[0].data
            assert type(dim) == int  # sanity check
            # crop the image into (dim, dim) square
            cropped = get_square(data, target, dim)
    return cropped


def get_square(data: np.array, center: tuple, dim: int) -> np.array:
    """
    Crops the FITS image to a (dim, dim) shaped numpy array, centered on
    the object of interest.

    For more info on the WCS for astropy & how it handles (x, y) math vs array
    convention: https://docs.astropy.org/en/stable/wcs/index.html
    https://docs.astropy.org/en/stable/api/astropy.wcs.utils.skycoord_to_pixel.html

    :param data: np.array of the actual image itself
    :param center: (columns, rows) - (x, y) in math perspective
    :param dim: int for square dimensions to return

    :return: np.array of shape: (dim, dim). Is the cropped image.
    """
    center = list(center)

    # Center[0] is column wise, so shape[1]
    if center[0] - dim // 2 < 0:
        center[0] = dim // 2
    if center[0] + dim // 2 > data.shape[1]:
        center[0] = data.shape[0]

    # Center[1] is row wise, so shape[0]
    if center[1] - dim // 2 < 0:
        center[1] = dim // 2
    if center[1] + dim // 2 > data.shape[0]:
        center[1] = data.shape[0]

    return data[center[1] - dim // 2 : center[1] + dim // 2, center[0] - dim // 2 : center[0] + dim // 2].copy()
