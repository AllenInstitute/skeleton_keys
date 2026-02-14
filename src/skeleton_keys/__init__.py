from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("skeleton_keys")
except PackageNotFoundError:
    # package is not installed
    __version__ = "uninstalled"
