from contextlib import contextmanager
import zipfile

import luigi
import pandas as pd
import astropy


def get_data_cube(observation: pd.Series) -> np.ndarray:
    """Big, big workhorse. Takes in a row of the tabular data corresponding to one observation"""

    bands = ["u", "g", "r", "i", "z"]
    files = []
    for b in bands:
        files.append(
            f"frame-{b}-{str(observation.run).zfill(6)}-{observation.camcol}-{str(observation.field).zfill(4)}.fits.bz2"
        )

    # Want to switch to loading in the dimensions from a config file
    data_cube = np.zeros((32, 32, 5), dtype=np.float64)
    for i, f in enumerate(files):
        data_cube[i] = get_cropped(observation, f)

    # datacube should be (32,32,5)
    return data_cube


def get_cropped(observation: pd.Series, fits_file: str) -> np.array:
    """
    Given a file name, returns the cropped image as np array centered on the observation
    Unzips the file, converts the RA & DEC of obj to pixel coords, centers on object & crops to 32x32.
    Also deletes the unzipped file, since all of the data unzipped would be almost 200GB.
    """
    # can I even set the extracted file name like that?
    extracted_path = os.path.join("data", "tmp", "data.fits")

    try:
        # unzip data
        with zipfile.ZipFile(fits_file, "r") as zip_ref:
            # not sure if that's the path I want to use or how I want to handle unzipping
            zip_ref.extractall(extracted_path)

        # center of the galaxy
        center = galactic_coord_to_pixel(observation, fits_file)
        with astropy.io.fits.open(extracted_path) as data:
            # check this logic, also don't want to hardcode the dimensions
            cropped = data[center + 16 : center + 16, center - 16 : center + 16].copy()

    finally:
        if os.path.exists(extracted_path):
            os.remove(extracted_path)

    # kinda want there to be an error if the cropped doesn't work
    return cropped


def galactic_coord_to_pixel(observation: pd.Series, fits: str) -> tuple:
    """
    Thanks to https://lukaswenzl.com/wold-coordinate-system-astropy/ for the reminders!

    TODO: make sure I have the right coordinate system units getting passed around

    observation: row in tabular data corresponding to observation
    fits: path to corresponding unzipped fits image
    """
    # ish?? Not sure what the unzipped would be like
    fits = os.path.join("data/tmp/data.fits")

    # make sure this is actually a context manager
    with astropy.io.fits.open(fits) as hdulist:
        # do stuff
        header = hdulist[0]
        wcs = astropy.wcs.WCS(header)
        coord = SkyCoord(ra=observation.ra, dec=observation.dec)
        target = astropy.wcs.utils.skycoord_to_pixel(coord, wcs)

    return target


###====================================================###


class FITSLocalTarget(luigi.LocalTarget):
    hdulist

    @contextmanager
    def open(self, mode="r", **kwargs):
        # data = hdulist[0].data.copy()
        try:
            hdulist = astropy.io.fits.open(filename)
            yield hdulist
        finally:
            hdulist.close()
