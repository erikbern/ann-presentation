import struct

def _get_vectors_plain(f):
    for line in f:
        items = line.strip().split()
        yield items[0], [float(x) for x in items[1:]]

def _get_vectors(fn):
    if fn.endswith('.gz'):
        for line in _get_vectors_plain(gzip.open(fn)):
            yield line

    elif fn.endswith('.bin'): # word2vec format
        f = open(fn)
        words, size = (int(x) for x in f.readline().strip().split())

        t = 'f' * size

        while True:
            pos = f.tell()
            buf = f.read(1024)
            if buf == '': return
            i = buf.index(' ')
            word = buf[:i]
            f.seek(pos + i + 1)

            vec = struct.unpack(t, f.read(4 * size))

            yield word, vec

    else: # Assume simple text format
        for line in _get_vectors_plain(open(fn)):
            yield line

def get_vectors(fn, n=float('inf')):
    i = 0
    for line in _get_vectors(fn):
        yield line
        i += 1
        if i >= n:
            break
