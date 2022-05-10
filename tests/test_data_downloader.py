from unittest import TestCase
import numpy as np
import luigi
import os

from final_project import data_downloader


class DownloaderTests(TestCase):
    def test_downloader(self):
        """TODO"""
        d = data_downloader.URLgenerator()
        # print(d)
        # luigi.build([data_downloader.URLgenerator()], local_scheduler=True)

    def test_check_urls(self, n_urls=15445):
        """Need to adjust, there won't be 50000 unique urls, only 15,445"""
        urls = np.genfromtxt("data/urls.txt", dtype=str)
        assert np.unique(urls).shape[0] == n_urls

    def test_check_downloaded(self):
        not_downloaded = "data/not_downloaded.txt"
        with open("data/urls.txt", mode="r") as urls, open(not_downloaded, mode="w") as missing:
            for url in urls:
                file_name = url[-31:-1]
                if not os.path.exists("data/images/" + file_name):
                    missing.write(url)
        print("Done")

    def test_check_corrupted(self):
        """Using astropy's file checking to make sure the files didn't
        get corrupted.
        TODO removed hard coded paths here."""
        urls = np.genfromtxt("data/urls.txt", dtype=str)
        for url in urls:
            file_name = url[-31:-1]
            path = "data/images/" + file_name
            with bz2.BZ2File(path, "rb") as file:
                with astropy.io.fits.open(path) as hdulist:
                    pass

    def test_tabular(self):
        """Need to do this to check what's not in the tabular ..."""
        not_downloaded = "data/not_downloaded.txt"
        # with open("data/urls.txt", mode="r") as urls, open(not_downloaded, mode="w") as missing:
        #     for url in urls:
        #         file_name = url[-31:-1]
        #         if not os.path.exists("data/images/" + file_name):
        #             missing.write(url)
        # print("Done")
