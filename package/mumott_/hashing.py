import logging
import hashlib
from _hashlib import HASH

import numpy as np

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _cf_hasher(hashing_function, item) -> None:
    """ Internal method for hashing floats and complex number,
        or arrays of them. """
    if not isinstance(item, np.ndarray):
        item = np.array(item)
    item = item.ravel()
    if item.dtype.kind == 'c':
        # frexp only works on reals
        item = np.concatenate((item.real, item.imag))
    mantissa, exponent = np.frexp(item)
    edge_cases = np.isclose(abs(mantissa), 1.0, atol=1e-6, rtol=1e-6)
    mantissa[edge_cases] = np.sign(mantissa[edge_cases]) * 0.5
    exponent[edge_cases] = exponent[edge_cases] + 1
    # Round mantissa for consistency
    hashing_function.update(mantissa.round(5))
    hashing_function.update(exponent)


def _array_hasher(hashing_function: HASH, item: NDArray) -> None:
    """ Internal method for hashing arrays, lists and tuples. """
    if type(item) in (list, tuple):
        item = np.array(item)
    if item.dtype.kind in ('v'):
        hashing_function.update(item)
    # kind is bytes, int, uint, string, unicode
    if item.dtype.kind in ('biuSU'):
        hashing_function.update(np.char.encode(item.astype(str), 'utf-8'))
    # kind is float or complex
    elif item.dtype.kind in ('fc'):
        _cf_hasher(hashing_function, item)
    # unknown data, possibly ragged array etc
    else:
        raise TypeError(f'Hash of dtype `object` is not deterministic, cannot hash {item}')


def _item_hasher(hashing_function: HASH, item) -> None:
    """ Internal method for hashing floats, integers and strings. """
    if item is None:
        return
    if np.array(item).dtype.kind == 'v':
        hashing_function.update(item)
    if np.array(item).dtype.kind in ('biuSU'):
        # Cast all ints, strings, etc to string and encode
        hashing_function.update(str(item).encode('utf-8'))
    elif np.array(item).dtype.kind in ('fc'):
        _cf_hasher(hashing_function, item)
    elif np.array(item).dtype.kind == 'O':
        raise TypeError(f'Cannot hash unknown object: {item}')


def _dict_hasher(hashing_function: HASH, item) -> None:
    """ Internal method for hashing dictionaries. """
    for key, value in item.items():
        hashing_function.update(key.encode('utf-8'))
        if isinstance(value, np.ndarray) or type(value) in (list, tuple):
            _array_hasher(hashing_function, value)
        else:
            _item_hasher(hashing_function, value)


def list_to_hash(list_to_hash: list, hashing_algorithm: str = 'blake2b') -> str:
    """
    Function which takes a list containing a set of objects and automatically
    generates a deterministic hash for them.

    Parameters
    ----------
    list_to_hash
        List of a set of objects of various types, see `notes` for a complete list.
    hashing_algorithm
        The hashing algorithm to use. Can be any algorithm name in
        ``hashlib.algorithms_available``. Default is ``'blake2b'``.

    Example
    -------
    The following code snippets illustrate hashing lists that will work, and ones
    that will not work.

    Works: A list of an integer, an array, a dictionary with valid types, and a None.

    >>> from mumott.core.hashing import list_to_hash
    >>> print(list_to_hash([1, np.array((1, 3, 5)), dict(val=1, string='abc'), None]))
    2a949c...

    Does not work: an array containing a ``None``, due to the ``dtype`` being ``object``.

    >>> print(list_to_hash([np.array([None])]))
    Traceback (most recent call last):
    ...
    TypeError: Hash of dtype `object` is not deterministic, cannot hash [None]

    Does not work: a generator expression, which is an unknown object.

    >>> print(list_to_hash([(a for a in [1, 2, 3])]))
    Traceback (most recent call last):
    ...
    TypeError: Cannot hash unknown object: <generator object...

    Notes
    -----
    ``float``-type objects are rounded to five significant digits in the mantissa before hashing.
    This is necessary to obtain semi-deterministic hashes that obey a subset of fuzzy equality
    for float comparison. There are edge cases where equality can fail due to
    rounding errors, but these should be extremely rare.

    Supported entry types in :attr:`list_to_hash`:
        ``int``
            Cast to string.
            Works along with similar ``numpy`` types.
        ``float``
            Mantissa rounded to five significant digits and concatenated with exponent.
            Works along with similar ``numpy`` types.
        ``complex``
            Real and imaginary parts concatenated and treated like ``float``.
            Works along with similar ``numpy`` types.
        ``str``
            Automatically given ``'utf-8'`` encoding.
        ``bytes``
            Cast to string.
        ``None``
            Ignored.
        ``np.ndarray``
            Provided ``dtype`` is not ``object``, hence arrays of ``None`` are not allowed.
        ``list``, ``tuple``
            Provided they can be cast to allowed, i.e. non-ragged ``np.ndarray``
        ``dict``
            Assuming entries are allowed types. Keys and entries are concatenated.
            If an entry is ``None``, the key is added to the hash while the entry is ignored.
    """
    hashing_function = hashlib.new(hashing_algorithm)
    for item in list_to_hash:
        if isinstance(item, np.ndarray) or type(item) in (list, tuple):
            _array_hasher(hashing_function, item)
        elif type(item) is dict:
            _dict_hasher(hashing_function, item)
        else:
            _item_hasher(hashing_function, item)
    return hashing_function.hexdigest()
