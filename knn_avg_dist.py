import numpy
from scipy.spatial import distance
import matplotlib.pyplot as plt

n = 10000

fs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ts = []
for f in fs:
    sums = numpy.zeros(3)
    n_iters = 100
    for i in xrange(n_iters):
        x = numpy.random.normal(0, 1, f)
        dists = [distance.euclidean(x, numpy.random.normal(0, 1, f)) for i in xrange(n)]
        dists.sort()
        print dists[0]
        sums += numpy.array([dists[0], dists[9], dists[-1]])

    ts.append(sums/n_iters)
    print f, sums/n_iters

fig, ax = plt.subplots()
ax.plot(fs, ts, 'x-', ms=10, mew=5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, 'both')
ax.legend(['Distance to nearest neighbor',
           'Distance to neighbor #10',
           'Distance to furthest neighbor'], loc=4)
ax.set_title('%d points from a normal distribution' % n)
ax.set_xlabel('Number of dimensions')
ax.set_ylabel('Euclidean distance')
plt.savefig('knn_avg_dist.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
