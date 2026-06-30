from abc import ABC, abstractmethod
from itertools import chain
from typing import NoReturn

from audiogear.data import AudioPipeline
from audiogear.utils.import_utils import _is_package_available


class PipelineStep(ABC):
    name: str = None
    type: str = None

    def __new__(cls, *args, **kwargs):
        """
            Checks if this block or its superclasses' dependencies are installed and raises an error otherwise.
        Args:
            *args:
            **kwargs:
        """
        required_dependencies = chain.from_iterable(getattr(t, "_requires_dependencies", []) for t in cls.mro())
        if required_dependencies:
            missing_dependencies: dict[str, str] = {}
            for dependency in required_dependencies:
                dependency = dependency if isinstance(dependency, tuple) else (dependency, dependency)
                package_name, pip_name = dependency
                if not _is_package_available(package_name):
                    missing_dependencies[package_name] = pip_name
            if missing_dependencies:
                _raise_error_for_missing_dependencies(cls.__name__, missing_dependencies)
        return super().__new__(cls)

    def __repr__(self):
        return f"{self.type}: {self.name}"

    @abstractmethod
    def run(self, data: AudioPipeline, rank: int =0, world_size: int = 1) -> AudioPipeline:
        if data:
            segments = data
        else:
            segments = []
        return segments

    def __call__(self, data: AudioPipeline = None, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        return self.run(data, rank, world_size)


def _raise_error_for_missing_dependencies(step_name: str, dependencies: dict[str, str]) -> NoReturn:
    dependencies = dict(sorted(dependencies.items()))
    package_names = list(dependencies)
    if len(dependencies) > 1:
        package_names = (
            f"{','.join('`' + package_name + '`' for package_name in package_names[:-1])} and `{package_names[-1]}`"
        )
    else:
        package_names = f"`{package_names[0]}`"
    raise ImportError(
        f"Please install {package_names} to use {step_name} (`pip install {' '.join(list(dependencies.values()))}`)."
    )
