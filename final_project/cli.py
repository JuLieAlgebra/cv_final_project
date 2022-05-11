import luigi

from final_project import data_downloader, preprocessing, salted, preprocessing_utils


def get_ranges(n_workers=14, n_obj=1000):
    """Helper function for kicking off the image processing tasks.

    :param n_workers: int
    :param n_obj: int
    :return: np.array
    """
    chunk = n_obj // n_workers
    ranges = [[i, i + chunk] for i in range(0, n_obj, chunk)]
    ranges[-1][1] = n_obj
    return ranges


def main():
    """Main function"""
    n_workers = 14
    n_obj = 4000
    ranges = get_ranges(n_workers=n_workers, n_obj=n_obj)

    luigi.build(
        [preprocessing.Preprocessing(lower=i[0], upper=i[1]) for i in ranges],
        local_scheduler=True,
        workers=n_workers,
    )
