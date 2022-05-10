import luigi

# from final_project import MasterTask
from final_project import data_downloader, preprocessing, salted, preprocessing_utils


def main():
    #########################################################

    n_workers = 12
    n_urls = 50000
    n_obj = 10000
    chunk_url = n_urls // n_workers
    chunk = n_obj // n_workers
    ranges = [[i, i + chunk] for i in range(0, n_obj, chunk)]
    # print("RANGES: ", ranges)
    # for i in ranges:
    #     print(i[0], i[1])
    # print([(i[0], i[1]) for i in ranges])
    # quit()
    ranges[-1][1] = n_obj
    print(ranges)

    # assert n_urls % n_workers == 0  # if this isn't an integer, I want an error

    # luigi.build(
    #     [data_downloader.ImageDownloader(lower=i, upper=i + chunk) for i in range(0, n_urls, chunk)],
    #     local_scheduler=True,
    #     workers=n_workers,
    # )

    luigi.build(
        [preprocessing.Preprocessing(lower=i[0], upper=i[1]) for i in ranges],
        local_scheduler=True,
        workers=n_workers,
    )

    ########################################################

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
