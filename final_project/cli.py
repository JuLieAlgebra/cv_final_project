import luigi

# from final_project import MasterTask
from final_project import data_downloader


def main():
    luigi.build([data_downloader.TabularDownloader()], local_scheduler=True)
    quit()
    n_workers = 10
    n_urls = 50000
    chunk = n_urls // n_workers
    assert n_urls % n_workers == 0  # if this isn't an integer, I want an error

    luigi.build(
        [data_downloader.ImageDownloader(lower=0, upper=2)],
        local_scheduler=True,
    )
    # luigi.build(
    #     [data_downloader.ImageDownloader(lower=i, upper=i + chunk) for i in range(0, n_urls, chunk)],
    #     local_scheduler=True,
    #     workers=n_workers,
    # )
    # task = data_downloader.Downloader
    # for param_name, param in sorted(task.get_params()):
    #     print(param_name, param._default)


#     luigi.build(
#         [
#             final_project.MasterTask(args) # something like that
#         ],
#         local_scheduler=True,
#         workers=4
#     )
