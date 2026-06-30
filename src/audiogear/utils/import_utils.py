import importlib.resources
from functools import lru_cache


@lru_cache
def _is_package_available(package_name):
    """

    Args:
      package_name:

    Returns:

    """
    return importlib.util.find_spec(package_name) is not None


# use for tests


# Used in tests


def is_boto3_available():
    return _is_package_available("boto3")


def is_s3fs_available():
    return _is_package_available("s3fs")


def is_moto_available():
    return _is_package_available("moto")
