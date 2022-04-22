import os
from contextlib import contextmanager

import luigi


class CSV(luigi.Task):
    csv_path = "data/final_paper_2018.csv"
    csv = luigi.Parameter(default=csv_path)

    def output(self):
        return luigi.LocalTarget(self.csv)


class URlgenerator(luigi.Task):
    url_filename = "urls.txt"
    df = pd.read_csv(self.inputs().path)  # ??

    def requires(self):
        """Requres luigi Target of downloaded csv tabular data"""
        # csv target
        return CSV()

    def output(self):
        """Outputs formatted urls for downloading"""
        return luigi.LocalTarget("data/urls.text")  # ??

    def run(self):
        """ """
        urls = []
        bands = ["u", "g", "r", "i", "z"]
        for i, row in self.df.iterrows():
            for band in bands:
                urls.append(
                    f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/ \
                            {row.rerun}/{row.run}/{row.camcol}/frame-{band}-{str(row.run).zfill(6)}\
                            -{row.camcol}-{str(row.field).zfill(4)}.fits.bz2"
                )

        with open(self.url_filename, mode="w") as f:
            for u in urls:
                f.write(u)


class Downloader(luigi.Task):
    url_input = self.input()
    file_names = self.get_filenames(url_input)  # something like that for url

    def requires(self):
        return URlgenerator()

    def output(self):
        """Right now this would do every single 50k image"""
        return [file for file in self.files]

    @contextmanager
    def download(self, stuff):
        tmp = "tmp.fits.bz2"
        try:
            os.system(f"wget -O {tmp} {urls[0]}")
            os.system(f"mv tmp.fits.bz2 frame-u-000756-2-0626.fits.bz2")
        # except WhateverExceptionIexpect:
        # alert me that it happened
        finally:
            if os.exists(tmp):
                os.remove(tmp)

    def run(self):
        """ """
        for file in self.files:
            with download(file) as d:
                pass
