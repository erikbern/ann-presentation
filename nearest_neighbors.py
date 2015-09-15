'''
Usage: python nearest_neighbors.py <data file> [<search k>]

(works like ./distance in word2vec)

* Data file can be
  1. A .bin file that is compatible with word2vec binary formats.
     There are pre-trained vectors to download at https://code.google.com/p/word2vec/
  2. A .gz file with the GloVe format (item then a list of floats in plaintext)
  3. A plain text file with the same format as above
* Dependencies: lmdb (pip install lmdb)
* Will be (very) slow first time because it creates big data structures, fast subsequent times
'''

import annoy
import lmdb
import os
import struct
import sys
import numpy

from util import get_vectors

if len(sys.argv) < 2:
    print __doc__
    raise

fn = sys.argv[1]
search_k = 100000
if len(sys.argv) > 2:
    search_k = int(sys.argv[2])

fn_annoy = fn + '.annoy'
fn_lmdb = fn + '.lmdb' # stores word <-> id mapping

word, vec = get_vectors(fn).next()
size = len(vec)
env = lmdb.open(fn_lmdb, map_size=int(1e9))

if not os.path.exists(fn_annoy) or not os.path.exists(fn_lmdb):
    i = 0
    a = annoy.AnnoyIndex(size)
    with env.begin(write=True) as txn:
        for word, vec in get_vectors(sys.argv[1]):
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
    for line in sys.stdin:
        if line.strip() == 'EXIT':
            break

        words = line.strip().split()
        ids = [int(txn.get('w' + word)[1:]) for word in words]
        vs = [a.get_item_vector(id) for id in ids]
        vs = [v / numpy.dot(v, v)**0.5 for v in vs]
        v = numpy.sum(vs, axis=0)

        for id, dist in zip(*a.get_nns_by_vector(v, 20, search_k, True)):
            if id not in ids:
                word = txn.get('i%d' % id)[1:]
                print '%50s\t%f' % (word, dist)

