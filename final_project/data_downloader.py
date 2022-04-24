import os
from contextlib import contextmanager
from typing import ContextManager

import luigi
import pandas as pd


class CSV(luigi.Task):
    csv_path = "data/final_paper_2018.csv"
    csv = luigi.Parameter(default=csv_path)

    def output(self):
        return luigi.LocalTarget(self.csv)


class URlgenerator(luigi.Task):
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
        df = pd.read_csv(self.inputs().path)
        urls = []
        bands = ["u", "g", "r", "i", "z"]
        for i, row in self.df.iterrows():
            for band in bands:
                urls.append(
                    f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/ \
                            {row.rerun}/{row.run}/{row.camcol}/frame-{band}-{str(row.run).zfill(6)}\
                            -{row.camcol}-{str(row.field).zfill(4)}.fits.bz2"
                )
        print("Success: Generated URLs")

        with self.output().open(mode='w') as f:
            for u in urls:
                f.write(u)


class Downloader(luigi.Task):
    # url_input = self.input()
    # file_names = self.get_filenames(url_input)  # something like that for url

    def requires(self):
        return URlgenerator()

    def output(self):
        """Right now this would do every single 50k image"""
        return {url: luigi.LocalTarget(url[url.find("frame"):])
                for url in self.inputs()}

    @contextmanager
    def download(self, url) -> ContextManager:
        tmp = f"{hash(url)}.fits.bz2"
        file_name = url[url.find("frame"):]
        try:
            os.system(f"wget -O data/{tmp} {urls[0]}")
            os.system(f"mv data/{tmp} data/{file_name}")
            yield file_name
        # except WhateverExceptionIexpect:
        # alert me that it happened
        finally:
            if os.exists(tmp):
                os.remove(tmp)

    def run(self):
        """ Downloads and moves the images from the SDSS atomically """
        with self.inputs().open('r') as urls:
            for url in urls:
                with download(url) as d:
                    print(f"Success: {d}")
