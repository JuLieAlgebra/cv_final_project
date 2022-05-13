import os
from contextlib import contextmanager
from typing import ContextManager
import bz2

import omegaconf
import luigi
from luigi.contrib.s3 import S3Target
import numpy as np
import pandas as pd
import astropy.io.fits

# for my own sanity
join = os.path.join
# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)


class FailedImageCheck(OSError):
    """ """

    pass


class SavedS3(luigi.ExternalTask):
    """Luigi Task to point to S3 target"""

    __version__ = "0.1.0"

    paths = omegaconf.OmegaConf.load(join(abs_path, "conf", "aws_paths.yaml"))
    data = luigi.Parameter(default="tabular")

    def output(self) -> S3Target:
        """Points to S3 tabular data target"""
        s3_path = self.paths[f"{self.data}_dir"] + self.paths[f"{self.data}_data"]
        return S3Target(s3_path, format=luigi.format.Nop)


class TabularDownloader(luigi.Task):
    """Luigi Task to download S3 target"""

    __version__ = "0.1.0"

    tabular_data = omegaconf.OmegaConf.load(join(abs_path, "conf", "aws_paths.yaml"))["tabular_data"]
    csv_path = join("data", f"{tabular_data}")
    csv = luigi.Parameter(default=csv_path)

    def requires(self) -> luigi.ExternalTask:
        """
        Luigi function to kick off the requirements for this task before running it - part of the DAG.
        This one needs the tabular file to exist on S3 before it can download it.
        """
        # string to grab path from yaml config
        return SavedS3("tabular")

    def output(self) -> luigi.LocalTarget:
        """
        Luigi function to understand what the 'I'm done' file output will be from this task so it can
        run the next in the sequence.
        """
        return luigi.LocalTarget(self.csv, format=luigi.format.Nop)

    def run(self) -> None:
        """Downloads the csv from S3"""
        with self.input().open("r") as f, self.output().open("w") as outfile:
            outfile.write(f.read())


class URLgenerator(luigi.Task):
    """Luigi task to generate the URLS from the tabular data file"""

    __version__ = "0.1.0"
    url_filename = luigi.Parameter(default=join("data", "urls.txt"))

    def requires(self):
        """Requres luigi Target of downloaded csv tabular data"""
        return TabularDownloader()

    def output(self):
        """Outputs formatted urls for downloading"""
        return luigi.LocalTarget(self.url_filename)

    def run(self):
        """Generates the URLs"""
        df = pd.read_csv(self.input().path)
        urls = []
        bands = ["u", "g", "r", "i", "z"]
        for i, row in df.iterrows():
            for band in bands:
                urls.append(
                    f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/{row.rerun}/{row.run}/{row.camcol}/frame-{band}-{str(row.run).zfill(6)}-{row.camcol}-{str(row.field).zfill(4)}.fits.bz2\n"
                )
        print("Success: Generated URLs")

        with self.output().open(mode="w") as f:
            for u in urls:
                f.write(u)


class ImageDownloader(luigi.Task):
    """
    Downloads all files in specified range of the urls.txt. Usually used with copies of itself
    with different parameters to span the whole range of urls to download.

    Example::
        n_workers = 10
        n_urls = 50000
        chunk = n_urls // n_workers

        luigi.build(
            [data_downloader.Downloader(lower=i, upper=i + chunk) for i in range(0, n_urls, chunk)],
            local_scheduler=True,
            workers=n_workers)
    """

    __version__ = "0.1.0"

    # range of downloaded files = [lower, upper]
    lower = luigi.IntParameter()
    upper = luigi.IntParameter()

    def requires(self):
        """
        Luigi function to kick off the requirements for this task before running it - part of the DAG.
        This one requires the file of urls to download to be successfully made before it can download the images.
        """
        return URLgenerator()

    def output(self):
        """Success file with range of files"""
        return luigi.LocalTarget(join("data", "images", f"_SUCCESS{self.lower}-{self.upper}"))

    @classmethod
    @contextmanager  # test this
    def download(cls, url) -> ContextManager:
        """Context manager for downloading FITS images from the SDSS

        :param url: string url to fits file on SDSS server
        :return: ContextManager"""

        # Generating temporary and final file names
        tmp = f"{hash(url)}.fits.bz2"
        # skipping the new line character at the end
        file_name = url[-31:-1]

        try:
            # If file already exists, don't download
            if os.path.exists(join("data", "images", f"{file_name}")):
                print("Skipping!!", file_name)
                yield file_name

            else:
                # download file to the tmp file name
                os.system(f"wget -O data/images/{tmp} {url}")
                # atomically move/rename the successful download to the final name
                os.system(f"mv data/images/{tmp} data/images/{file_name}")
                yield file_name
        finally:
            # This is messy to leave here, but I wanted to illustrate my thought process
            # This does check that the file isn't corrupted, which becomes a problem in the
            # pre-processing steps, but the data checking does take hours to run. I modified this
            # idea into a random check of a dozen files as part of my testing pipeline (which is not
            # designed for CI)

            # # If it doesn't exist, then something went wrong downloading and we're
            # # going to delete the temp.
            # if os.path.exists(join("data", "images", f"{file_name}")):
            #     # check image
            #     if cls.bad_file(file_name):
            #         os.remove(join("data", "images", f"{file_name}"))
            #         print("Got a bad one: ", file_name)
            #         cls.download(url)

            # Cleanup, if tmp still exists
            if os.path.exists(tmp):
                os.remove(tmp)

            if not os.path.exists(join("data", "images", f"{file_name}")):
                # didn't download, try again
                # TODO need to write a breakout for this recursion in case it just
                #      can't do it
                print("Trying again")
                cls.download(url)

    @classmethod
    def bad_file(cls, file_name: str) -> bool:
        """Using astropy's file checking to make sure the files didn't
        get corrupted.
        TODO removed hard coded paths here.

        :param file_name: string of the file to be checked, excludes directory
        :return: bool"""
        path = join("data", "images", f"{file_name}")
        try:
            with bz2.BZ2File(path, "rb") as file:
                with astropy.io.fits.open(path) as hdulist:
                    return False
        except OSError:
            return True
        # if we've gotten here.... something happened
        raise FailedImageCheck(f"What happened?: {file_name}")

    def run(self):
        """Downloads and moves the images from the SDSS atomically"""
        with self.input().open("r") as urls:
            for i, url in enumerate(urls):
                # only downloads the section of urls that have been passed in
                # as Luigi parameter
                if i >= self.lower and i <= self.upper:
                    # using our context manager to download the urls
                    with self.download(url) as d:
                        print(f"Success: {d}")

        # writing the success file
        with self.output().open("w") as outfile:
            outfile.write(f"Success: {self.lower}-{self.upper}")
