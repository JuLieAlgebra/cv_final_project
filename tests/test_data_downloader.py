from unittest import TestCase
from final_project import data_downloader
import luigi


class DownloaderTests(TestCase):
    def test_downloader(self):
        d = data_downloader.URLgenerator()
        # print(d)
        # luigi.build([data_downloader.URLgenerator()], local_scheduler=True)

    def test_check_urls(self, n_urls=50000):
        """Had issue during development that resulted in downloading duplicate files"""
        urls = np.genfromtxt("data/urls.txt", dtype=str)
        assert len(np.unique(urls)[0]) == n_urls

    def test_check_downloaded(self):
        not_downloaded = "data/not_downloaded.txt"
        with open("data/urls.txt", mode="r") as urls, open(not_downloaded, mode="w") as missing:
            for url in urls:
                file_name = url[-31:-1]
                if not os.path.exists("data/images/" + file_name):
                    missing.write(url)
        print("Done")
