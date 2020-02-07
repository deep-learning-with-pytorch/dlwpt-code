import collections
import copy
import datetime
import gc
import time

# import torch
import numpy as np

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module


# class dotdict(dict):
#     '''dict where key can be access as attribute d.key -> d[key]'''
#     @classmethod
#     def deep(cls, dic_obj):
#         '''Initialize from dict with deep conversion'''
#         return cls(dic_obj).deepConvert()
#
#     def __getattr__(self, attr):
#         if attr in self:
#             return self[attr]
#         log.error(sorted(self.keys()))
#         raise AttributeError(attr)
#         #return self.get(attr, None)
#     __setattr__= dict.__setitem__
#     __delattr__= dict.__delitem__
#
#
#     def __copy__(self):
#         return dotdict(self)
#
#     def __deepcopy__(self, memo):
#         new_dict = dotdict()
#         for k, v in self.items():
#             new_dict[k] = copy.deepcopy(v, memo)
#         return new_dict
#
#     # pylint: disable=multiple-statements
#     def __getstate__(self): return self.__dict__
#     def __setstate__(self, d): self.__dict__.update(d)
#
#     def deepConvert(self):
#         '''Convert all dicts at all tree levels into dotdict'''
#         for k, v in self.items():
#             if type(v) is dict: # pylint: disable=unidiomatic-typecheck
#                 self[k] = dotdict(v)
#                 self[k].deepConvert()
#             try: # try enumerable types
#                 for m, x in enumerate(v):
#                     if type(x) is dict: # pylint: disable=unidiomatic-typecheck
#                         x = dotdict(x)
#                         x.deepConvert()
#                         v[m] = x#

#             except TypeError:
#                 pass
#         return self
#
#     def copy(self):
#         # override dict.copy()
#         return dotdict(self)


def prhist(ary, prefix_str=None, **kwargs):
    if prefix_str is None:
        prefix_str = ''
    else:
        prefix_str += ' '

    count_ary, bins_ary = np.histogram(ary, **kwargs)
    for i in range(count_ary.shape[0]):
        print("{}{:-8.2f}".format(prefix_str, bins_ary[i]), "{:-10}".format(count_ary[i]))
    print("{}{:-8.2f}".format(prefix_str, bins_ary[-1]))

# def dumpCuda():
#     # small_count = 0
#     total_bytes = 0
#     size2count_dict = collections.defaultdict(int)
#     size2bytes_dict = {}
#     for obj in gc.get_objects():
#         if isinstance(obj, torch.cuda._CudaBase):
#             nbytes = 4
#             for n in obj.size():
#                 nbytes *= n
#
#             size2count_dict[tuple([obj.get_device()] + list(obj.size()))] += 1
#             size2bytes_dict[tuple([obj.get_device()] + list(obj.size()))] = nbytes
#
#             total_bytes += nbytes
#
#     # print(small_count, "tensors equal to or less than than 16 bytes")
#     for size, count in sorted(size2count_dict.items(), key=lambda sc: (size2bytes_dict[sc[0]] * sc[1], sc[1], sc[0])):
#         print('{:4}x'.format(count), '{:10,}'.format(size2bytes_dict[size]), size)
#     print('{:10,}'.format(total_bytes), "total bytes")


def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

#
# try:
#     import matplotlib
#     matplotlib.use('agg', warn=False)
#
#     import matplotlib.pyplot as plt
#     # matplotlib color maps
#     cdict = {'red':   ((0.0,  1.0, 1.0),
#                        # (0.5,  1.0, 1.0),
#                        (1.0,  1.0, 1.0)),
#
#              'green': ((0.0,  0.0, 0.0),
#                        (0.5,  0.0, 0.0),
#                        (1.0,  0.5, 0.5)),
#
#              'blue':  ((0.0,  0.0, 0.0),
#                        # (0.5,  0.5, 0.5),
#                        # (0.75, 0.0, 0.0),
#                        (1.0,  0.0, 0.0)),
#
#              'alpha':  ((0.0, 0.0, 0.0),
#                        (0.75, 0.5, 0.5),
#                        (1.0,  0.5, 0.5))}
#
#     plt.register_cmap(name='mask', data=cdict)
#
#     cdict = {'red':   ((0.0,  0.0, 0.0),
#                        (0.25,  1.0, 1.0),
#                        (1.0,  1.0, 1.0)),
#
#              'green': ((0.0,  1.0, 1.0),
#                        (0.25,  1.0, 1.0),
#                        (0.5, 0.0, 0.0),
#                        (1.0,  0.0, 0.0)),
#
#              'blue':  ((0.0,  0.0, 0.0),
#                        # (0.5,  0.5, 0.5),
#                        # (0.75, 0.0, 0.0),
#                        (1.0,  0.0, 0.0)),
#
#              'alpha':  ((0.0, 0.15, 0.15),
#                        (0.5,  0.3, 0.3),
#                        (0.8,  0.0, 0.0),
#                        (1.0,  0.0, 0.0))}
#
#     plt.register_cmap(name='maskinvert', data=cdict)
# except ImportError:
#     pass
