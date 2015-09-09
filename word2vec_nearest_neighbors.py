'''
Usage: python word2vec_distance.py <data file> <word>

* Data file has to be a .bin file that is compatible with word2vec binary formats.
* There are pre-trained vectors to download at https://code.google.com/p/word2vec/
* Dependencies: lmdb (pip install lmdb)
* Will be (very) slow first time because it creates big data structures, fast subsequent times
'''

import annoy
import lmdb
import os
import struct
import sys

def read_word2vec_format(fn):
    f = open(fn)
    # words, size = struct.unpack('QQ', f.read(16))
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

try:
    fn, word1 = sys.argv[1:]
except:
    print __doc__
    raise

fn_annoy = fn + '.annoy'
fn_lmdb = fn + '.lmdb' # stores word <-> id mapping

word, vec = read_word2vec_format(sys.argv[1]).next()
size = len(vec)
env = lmdb.open(fn_lmdb, map_size=int(1e9))

if not os.path.exists(fn_annoy) or not os.path.exists(fn_lmdb):
    i = 0
    a = annoy.AnnoyIndex(size)
    with env.begin(write=True) as txn:
        for word, vec in read_word2vec_format(sys.argv[1]):
            a.add_item(i, vec)
            id = 'i%d' % i
            word = 'w' + word
            txn.put(id, word)
            txn.put(word, id)
            i += 1
            if i % 1000 == 0:
                print i, '...'

    a.build(40)
    a.save(fn_annoy)

a = annoy.AnnoyIndex(size)
a.load(fn_annoy)

with env.begin() as txn:
    word = txn.get('w' + word1)
    id = int(word[1:])

    for id2 in a.get_nns_by_item(id, 10, 1000):
        word2 = txn.get('i%d' % id2)[1:]
        print word2

