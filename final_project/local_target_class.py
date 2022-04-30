from contextlib import contextmanager
import luigi
import astropy.io


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


def get_fits_data(filename: str) -> np.ndarray:
    """
    Returns image data (2D numpy array) of file.

    Parameters
    ----------
    filename: string of fits file name including path to object

    Returns
    -------
    data: 2D numpy array
    """
    if filename[-5:] != ".fits":
        filename += filename + ".fits"

    # opening file and extracting data
    hdulist = astropy.io.fits.open(filename)
    data = hdulist[0].data.copy()
    hdulist.close()

    return data


def get_data_stack(file_list: List[str]) -> np.ndarray:
    """
    Takes in list of fits files
    Each fits file must have image of the same shape

    First element of the data stack indexes into a fits image
    Think of the data stack as a literal stack of papers, with
    each paper as a fits image. Computing along axis 0 will go
    pixel-wise through each image, like an arrow stabbing the
    stack of papers.

    Returns 3D "cube" of fits image data
    If you only have one fits file, use get_fits_data instead

    Parameters
    ----------
    file_list: list of strings of fits file names

    Returns
    -------
    data_stack: 3D numpy array
    """
    data_slice_shape = np.shape(get_fits_data(file_list[0]))
    data_stack = np.zeros((len(file_list), data_slice_shape[0], data_slice_shape[1]))

    for index, file in enumerate(file_list):
        data_stack[index] = get_fits_data(file)

    return data_stack
