import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import BytesType, MODE_BINARY, BytesIO

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :return: (size, mode, filename, value) tuple for Cache table
        """
        # pylint: disable=unidiomatic-typecheck
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False

            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            for offset in range(0, len(value), 2**30):
                gz_file.write(value[offset:offset+2**30])
            gz_file.close()

            value = str_io.getvalue()

        return super(GzipDisk, self).store(value, read)


    def fetch(self, mode, filename, value, read):
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        value = super(GzipDisk, self).fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = BytesIO()

            while True:
                uncompressed_data = gz_file.read(2**30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break

            value = read_csio.getvalue()

        return value

def getCache(scope_str):
    return FanoutCache('data-unversioned/cache/' + scope_str,
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11,
                       # disk_min_file_size=2**20,
                       )

# def disk_cache(base_path, memsize=2):
#     def disk_cache_decorator(f):
#         @functools.wraps(f)
#         def wrapper(*args, **kwargs):
#             args_str = repr(args) + repr(sorted(kwargs.items()))
#             file_str = hashlib.md5(args_str.encode('utf8')).hexdigest()
#
#             cache_path = os.path.join(base_path, f.__name__, file_str + '.pkl.gz')
#
#             if not os.path.exists(os.path.dirname(cache_path)):
#                 os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#
#             if os.path.exists(cache_path):
#                 return pickle_loadgz(cache_path)
#             else:
#                 ret = f(*args, **kwargs)
#                 pickle_dumpgz(cache_path, ret)
#                 return ret
#
#         return wrapper
#
#     return disk_cache_decorator
#
#
# def pickle_dumpgz(file_path, obj):
#     log.debug("Writing {}".format(file_path))
#     with open(file_path, 'wb') as file_obj:
#         with gzip.GzipFile(mode='wb', compresslevel=1, fileobj=file_obj) as gz_file:
#             pickle.dump(obj, gz_file, pickle.HIGHEST_PROTOCOL)
#
#
# def pickle_loadgz(file_path):
#     log.debug("Reading {}".format(file_path))
#     with open(file_path, 'rb') as file_obj:
#         with gzip.GzipFile(mode='rb', fileobj=file_obj) as gz_file:
#             return pickle.load(gz_file)
#
#
# def dtpath(dt=None):
#     if dt is None:
#         dt = datetime.datetime.now()
#
#     return str(dt).rsplit('.', 1)[0].replace(' ', '--').replace(':', '.')
#
#
# def safepath(s):
#     s = s.replace(' ', '_')
#     return re.sub('[^A-Za-z0-9_.-]', '', s)
