from luigi import LocalTarget


class SaltedOutput:
    def __init__(
        self,
        file_pattern="{task.__class__.__name__}-{file}{salt}{self.ext}",
        ext=".csv",
        target_class=LocalTarget,
        file="" ** target_kwargs,
    ):
        file_pattern = file_pattern.format(salt=self.get_salted_version())
        self.file = file  # file is string of reg output
        self.file_pattern = file_pattern
        self.ext = ext
        self.target_class = target_class
        self.target_kwargs = target_kwargs

    def __get__(self, task: Task, cls):
        return partial(self.__call__, task)

    def __call__(self, task: Task) -> Target:
        # modified file pattern to be more the lecture from March 3rd
        return self.target_class(self.file_pattern.format(task=task, self=self))

    def get_salted_version(self, task: Task) -> str:
        """
        Rough version of Prof. Gorlin's implementation.
        """
        salt = ""
        # sorting the requirements as suggested to increase salt stability
        for req in sorted(flatten(task.requires())):
            salt += self.get_salted_version(req)

        salt += task.__class__.__name__ + task.__version__

        salt += "".join(
            [
                "{}={}".format(param_name, repr(task.param_kwargs[param_name]))
                for param_name, param in sorted(task.get_params())
            ]
        )

        return sha256(salt.encode()).hexdigest()[:10]
