from urllib import urlretrieve
import gzip
import numpy
import os
import scipy.linalg
import matplotlib.pyplot as plt

f = 200

fn = 'glove-%d.gz' % f
if not os.path.exists(fn):
    url = 'http://www-nlp.stanford.edu/data/glove.twitter.27B.%dd.txt.gz' % f
    print 'downloading', url, '->', fn
    urlretrieve(url, fn)

print 'reading...'
data = []
for line in gzip.open(fn, 'rb'):
    v = numpy.array([float(x) for x in line.strip().split()[1:]])
    data.append(v)
    if len(data) == 100000: break

data = numpy.array(data)
print data.shape

s = scipy.linalg.svd(data, compute_uv=False)

c = (numpy.cumsum(s**2) / numpy.dot(s, s))**0.5
fig, ax = plt.subplots()
ax.plot(c, 'r')
ax.set_ylabel('Proportion of space explained')
ax.set_xlabel('Truncated number of dimensions')
ax.set_title('SVD of GloVe f=200 dataset')
ax.grid(True, 'both')
plt.savefig('svd.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
