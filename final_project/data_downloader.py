import os
from contextlib import contextmanager
from typing import ContextManager

import omegaconf
import luigi
from luigi.contrib.s3 import S3Target
import numpy as np
import pandas as pd


class SavedS3(luigi.ExternalTask):
    """Luigi Task to point to S3 target"""

    paths = omegaconf.OmegaConf.load("final_project/conf/aws_paths.yaml")
    data = luigi.Parameter(default="tabular")

    def output(self) -> S3Target:
        s3_path = self.paths[f"{self.data}_dir"] + self.paths[f"{self.data}_data"]
        return S3Target(s3_path, format=luigi.format.Nop)


class TabularDownloader(luigi.Task):
    """Luigi Task to download S3 target"""

    tabular_data = omegaconf.OmegaConf.load("final_project/conf/aws_paths.yaml")["tabular_data"]
    csv_path = f"data/{tabular_data}"
    csv = luigi.Parameter(default=csv_path)

    def requires(self) -> luigi.ExternalTask:
        # string to grab path from yaml config
        return SavedS3("tabular")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.csv, format=luigi.format.Nop)

    def run(self) -> None:
        """Downloads the csv from S3"""
        with self.input().open("r") as f, self.output().open("w") as outfile:
            outfile.write(f.read())


class URLgenerator(luigi.Task):
    url_filename = luigi.Parameter(default="data/urls.txt")

    def requires(self):
        """Requres luigi Target of downloaded csv tabular data"""
        return TabularDownloader()

    def output(self):
        """Outputs formatted urls for downloading"""
        return luigi.LocalTarget(self.url_filename)  # ??

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

    Example usage:
        n_workers = 10
        n_urls = 50000
        chunk = n_urls // n_workers

        luigi.build(
            [data_downloader.Downloader(lower=i, upper=i + chunk) for i in range(0, n_urls, chunk)],
            local_scheduler=True,
            workers=n_workers)
    """

    # range of downloaded files = [lower, upper]
    lower = luigi.IntParameter()
    upper = luigi.IntParameter()

    def requires(self):
        return URLgenerator()

    def output(self):
        """Success file with range of files"""
        return luigi.LocalTarget(f"data/images/_SUCCESS{self.lower}-{self.upper}")

    @contextmanager
    def download(self, url) -> ContextManager:
        """Context manager for downloading FITS images from the SDSS"""

        # Generating temporary and final file names
        tmp = f"{hash(url)}.fits.bz2"
        # skipping the new line character at the end
        file_name = url[-31:-1]

        try:
            # If file already exists, don't download
            if os.path.exists(f"data/images/{file_name}"):
                print("Skipping!!", file_name)
                yield file_name

            else:
                # download file to the tmp file name
                os.system(f"wget -O data/images/{tmp} {url}")
                # atomically move/rename the successful download to the final name
                os.system(f"mv data/images/{tmp} data/images/{file_name}")
                yield file_name
        finally:
            # Cleanup, if tmp stil
            if os.path.exists(tmp):
                os.remove(tmp)

    def run(self):
        """Downloads and moves the images from the SDSS atomically"""
        with self.input().open("r") as urls:
            for i, url in enumerate(urls):
                # Ugly, want to redo, but functional
                # only downloads the section of urls that have been passed in
                # as Luigi parameter
                if i >= self.lower and i <= self.upper:
                    # using our context manager to download the urls
                    with self.download(url) as d:
                        print(f"Success: {d}")

        # writing the success file
        with self.output().open("w") as outfile:
            outfile.write(f"Success: {self.lower}-{self.upper}")
