from final_project import data_downloader, preprocessing, salted, preprocessing_utils
from unittest import TestCase
import tempfile
import luigi


class TestSalted(TestCase):
    def test_salted(self):
        """Tests basic functionality and that changing parameters
        results in new salt"""
        with tempfile.TemporaryDirectory() as tmp:

            class Requirement(luigi.Task):
                __version__ = "1.0"

                def output(self):
                    return luigi.LocalTarget("requirement.txt")

            class myTask(luigi.Task):
                __version__ = "1.0"
                param = luigi.Parameter()

                def output(self):
                    return luigi.LocalTarget("test.txt")

                def requires(self):
                    return Requirement()

            task = myTask("arg")
            salt = salted.get_salted_version(task)
            task.__version__ = "1.1"
            salt_update = salted.get_salted_version(task)

            assert salt != salt_update
