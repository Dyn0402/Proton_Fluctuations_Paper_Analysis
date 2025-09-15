#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 18 10:40 PM 2021
Created in PyCharm
Created as QGP_Scripts/istarmap

@author: Dylan Neff, dylan
"""

# istarmap.py for Python 3.7+
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap that returns an iterator instead of a list.
    Compatible with Python 3.7+ and 3.8+.
    """
    # In 3.7, there is no _check_running; instead use _state
    if hasattr(self, "_check_running"):
        self._check_running()
    else:
        if self._state != mpp.RUN:
            raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError("Chunksize must be >= 1, not {0:n}".format(chunksize))

    # Python 3.8+ has _get_tasks, Python 3.7 does not
    if hasattr(mpp.Pool, "_get_tasks"):
        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    else:
        # Backport task batching for 3.7
        def gen_batches():
            it = iter(iterable)
            while True:
                batch = []
                try:
                    for _ in range(chunksize):
                        batch.append(next(it))
                except StopIteration:
                    if batch:
                        yield batch
                    break
                yield batch

        task_batches = (((func, args), {}) for batch in gen_batches() for args in batch)

    result = mpp.IMapIterator(self)
    self._taskqueue.put((
        self._guarded_task_generation(result._job,
                                      mpp.starmapstar,
                                      task_batches),
        result._set_length
    ))
    return (item for chunk in result for item in chunk)


# Patch Pool
mpp.Pool.istarmap = istarmap


# istarmap.py for Python 3.8+
# import multiprocessing.pool as mpp
#
#
# def istarmap(self, func, iterable, chunksize=1):
#     """starmap-version of imap
#     """
#     self._check_running()
#     if chunksize < 1:
#         raise ValueError(
#             "Chunksize must be 1+, not {0:n}".format(
#                 chunksize))
#
#     task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
#     result = mpp.IMapIterator(self)
#     self._taskqueue.put(
#         (
#             self._guarded_task_generation(result._job,
#                                           mpp.starmapstar,
#                                           task_batches),
#             result._set_length
#         ))
#     return (item for chunk in result for item in chunk)
#
#
# mpp.Pool.istarmap = istarmap