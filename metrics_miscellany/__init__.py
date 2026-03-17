from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metrics-miscellany")
except PackageNotFoundError:
    __version__ = "0.0.0"

from . import estimators
from . import tests
from . import utils
from . import order_statistics
from datamat import DataMat, DataVec
from . import kernel_methods
import datamat
