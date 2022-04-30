import os
from contextlib import contextmanager
from typing import ContextManager

import luigi
import numpy as np
import pandas as pd


class CSV(luigi.Task):
    # csv_path = "data/final_paper_2018.csv"
    csv_path = "data/tenthousand_julieannabacon.csv"
    csv = luigi.Parameter(default=csv_path)

    def output(self):
        return luigi.LocalTarget(self.csv)


class URLgenerator(luigi.Task):
    url_filename = luigi.Parameter(default="data/urls.txt")

    def requires(self):
        """Requres luigi Target of downloaded csv tabular data"""
        # csv target
        return CSV()

    def output(self):
        """Outputs formatted urls for downloading"""
        return luigi.LocalTarget(self.url_filename)  # ??

    def run(self):
        """ """
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


class Downloader(luigi.Task):
    # range of downloaded files = [lower, upper]
    lower = luigi.IntParameter()
    upper = luigi.IntParameter()

    def requires(self):
        return URLgenerator()

    def output(self):
        """Right now this would do every single 50k image"""
        # Not sure...............................
        # return {url: luigi.LocalTarget(url[url.find("frame"):])
        #         for url in self.input()}
        return luigi.LocalTarget(f"data/images/_SUCCESS{self.lower}-{self.upper}")

    @contextmanager
    def download(self, url) -> ContextManager:
        # Generating temporary and final file names
        tmp = f"{hash(url)}.fits.bz2"
        # skipping the new line character at the end
        file_name = url[-31:-1]

        # If file already exists, don't download
        if os.path.exists(f"data/images/{file_name}"):
            print("Skipping!!", file_name)
            return

        try:
            # download file to the tmp file name
            os.system(f"wget -O data/images/{tmp} {url}")
            # atomically move/rename the successful download to the final name
            os.system(f"mv data/images/{tmp} data/images/{file_name}")
            yield file_name
        # except WhateverExceptionIexpect: TODO
        # alert me that it happened
        finally:
            # Cleanup, if tmp stil
            if os.path.exists(tmp):
                os.remove(tmp)

    def run(self):
        """Downloads and moves the images from the SDSS atomically"""
        with self.input().open("r") as urls:
            for i, url in enumerate(urls):
                # Very ugly, want to redo, but functional
                # only downloads the section of urls that have been passed in
                # as Luigi parameter
                if i >= self.lower and i <= self.upper:
                    # using our context manager to download the urls
                    with self.download(url) as d:
                        print(f"Success: {d}")

        # writing the success file
        with self.output().open("w") as outfile:
            outfile.write(f"Success: {self.lower}-{self.upper}")


def check_urls(n_urls=50000):
    urls = np.genfromtxt("data/urls.txt", dtype=str)
    print(urls.shape)
    print(len(np.unique(urls)))
    assert len(np.unique(urls)[0]) == n_urls


def check_downloaded():
    not_downloaded = "data/not_downloaded.txt"
    with open("data/urls.txt", mode="r") as urls, open(not_downloaded, mode="w") as missing:
        for url in urls:
            file_name = url[-31:-1]
            if not os.path.exists("data/images/" + file_name):
                missing.write(url)
    print("Done")
