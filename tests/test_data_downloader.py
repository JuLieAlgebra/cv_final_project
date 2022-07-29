from unittest import TestCase
import os
import bz2

import astropy.io
import numpy as np
import luigi

from final_project import data_downloader


class DownloaderTests(TestCase):
    def test_check_downloaded(self):
        """ Ensure that what was not downloaded once is now downloaded. """
        not_downloaded = "data/not_downloaded.txt"
        with open("data/urls.txt", mode="r") as urls, open(not_downloaded, mode="w") as missing:
            for url in urls:
                file_name = url[-31:-1]
                if not os.path.exists("data/images/" + file_name):
                    missing.write(url)
        print("Done")

    def test_check_corrupted(self, check=10):
        """
        Using astropy's file checking to make sure the files didn't
        get corrupted.
        TODO removed hard coded paths here.

        This will take 8 hours to check every single file, so I randomly check (no set seed) 10 files.
        """
        urls = np.genfromtxt("data/urls.txt", dtype=str)
        to_check = np.random.choice(urls, size=check, replace=False)
        checked = {}
        for url in to_check:
            file_name = url[-30:]
            path = os.path.join("data", "images", file_name)
            if not checked.get(file_name):
                with bz2.BZ2File(path, "rb") as file:
                    with astropy.io.fits.open(file) as hdulist:
                        checked[file_name] = True

    def test_tabular(self):
        """
        Need to do this to check what's not in the tabular ...
        Maybe need to join the urls with the dataframe from tabular, have url be a part of the data
        """
        not_downloaded = "data/not_downloaded.txt"
        with open("data/urls.txt", mode="r") as urls, open(not_downloaded, mode="w") as missing:
            for url in urls:
                file_name = url[-31:-1]
                if not os.path.exists("data/images/" + file_name):
                    missing.write(url)
        print("Done")
