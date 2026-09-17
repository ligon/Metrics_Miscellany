from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metrics-miscellany")
except PackageNotFoundError:
    __version__ = "0.0.0"

import datamat
from datamat import DataMat, DataVec

from . import estimators
from . import kernel_methods
from . import order_statistics
from . import tests
from . import utils

# Re-exported for `from metrics_miscellany import ...`; named here so that
# they read as the package's public surface rather than as unused imports.
__all__ = [
    "DataMat",
    "DataVec",
    "datamat",
    "estimators",
    "kernel_methods",
    "order_statistics",
    "tests",
    "utils",
    "__version__",
]
