import luigi
# from final_project import MasterTask
from final_project import data_downloader


def main():
    # luigi.build([data_downloader.URLgenerator()], local_scheduler=True)
    luigi.build([data_downloader.Downloader()], local_scheduler=True)
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

