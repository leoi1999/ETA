"""Shared helpers for loading serialized modality features."""

import sys


def install_numpy_pickle_compat():
    """Allow NumPy 2 pickles to load in NumPy 1 runtime environments."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
        import numpy.core._multiarray_umath as numpy_umath
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)
    sys.modules.setdefault("numpy._core._multiarray_umath", numpy_umath)
