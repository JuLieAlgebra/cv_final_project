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


def get_data_cube(observation: pd.Series) -> np.ndarray:
    """Workhorse. Takes in a row of the tabular data corresponding to one observation"""
    dim = int(omegaconf.OmegaConf.load(os.path.join("final_project", "conf", "preprocessing.yaml"))["dim"])

    bands = ["u", "g", "r", "i", "z"]
    files = []
    for b in bands:
        files.append(
            f"frame-{b}-{str(observation.run).zfill(6)}-{observation.camcol}-{str(observation.field).zfill(4)}.fits.bz2"
        )

    data_cube = np.zeros((5, dim, dim), dtype=np.float64)
    for i, f in enumerate(files):
        data_cube[i] = get_cropped(observation, f, dim, bands[i])

    # datacube should be (5, dim, dim)
    return data_cube


# TODO get rid of this function
def get_cropped(observation: pd.Series, fits_file: str, dim: int, band: str) -> np.array:
    """
    Given a file name, returns the cropped image as np array centered on the observation
    Unzips the file, converts the RA & DEC of obj to pixel coords, centers on object & crops to 32x32.
    Also deletes the unzipped file, since all of the data unzipped would be almost 200GB.
    """
    # config path loading
    local_paths = omegaconf.OmegaConf.load(os.path.join("final_project", "conf", "local_paths.yaml"))
    fits_file_path = os.path.join(local_paths["data"], local_paths["images"], fits_file)

    with bz2.BZ2File(fits_file_path, "rb") as file:
        # center of the galaxy
        return galactic_coord_to_pixel(observation=observation, fits=file, dim=dim)


def galactic_coord_to_pixel(observation: pd.Series, fits: str, dim: int) -> tuple:
    """
    Thanks to https://lukaswenzl.com/wold-coordinate-system-astropy/ for the reminders!

    observation: row in tabular data corresponding to observation
    fits: path to corresponding unzipped fits image
    """

    with astropy.io.fits.open(fits) as hdulist:
        # do stuff
        header = hdulist[0].header
        # create world coordinate system from header
        wcs = astropy.wcs.WCS(header)
        # get a coordinate object of observation
        coord = astropy.coordinates.SkyCoord(ra=observation.ra, dec=observation.dec, unit='deg')
        # pixel coordinates of center of observation
        target = astropy.wcs.utils.skycoord_to_pixel(coord, wcs)
        target = (int(target[0]), int(target[1]))
        # actual image
        data = hdulist[0].data
        #
        assert type(dim) == int
        # crop the image into (dim, dim) square
        cropped = get_square(data, target, dim)
    return cropped


def get_square(data: np.array, center: tuple, dim: int) -> np.array:
    """ TODO: check logic and check the return for skycoord to pixel?? x and y or y and x?

    center: (columns, rows) - (x, y) in math perspective
    https://docs.astropy.org/en/stable/wcs/index.html
    https://docs.astropy.org/en/stable/api/astropy.wcs.utils.skycoord_to_pixel.html

    returns: (dim, dim) square
    """
    center = list(center)

    # Center[0] is column wise, so shape[1]
    if center[0] - dim//2 < 0:
        # print("case 1: ")
        center[0] = dim//2
    if center[0] + dim//2 > data.shape[1]:
        # print("case 2: ")
        center[0] = data.shape[0]
    # Center[1] is row wise, so shape[0]
    if center[1] - dim//2 < 0:
        # print("case 3: ")
        center[1] = dim//2
    if center[1] + dim//2 > data.shape[0]:
        # print("case 4: ")
        center[1] = data.shape[0]

    return data[center[1] - dim // 2 : center[1] + dim // 2, center[0] - dim // 2 : center[0] + dim // 2].copy()