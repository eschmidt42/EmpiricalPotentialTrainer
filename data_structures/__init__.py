from .inhouse_formats import supercell

__all__ = ["supercell"]

import warnings
def deprecate(fun):
    warnings.warn("Deprecation warning - this function ({} > {}) will be removed in future updates!".format(fun.__module__,fun.__name__))
    return fun