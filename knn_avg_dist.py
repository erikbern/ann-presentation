import numpy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random
import sys
import gzip

n = 10000

def get_gaussian(n, f):
    return [numpy.random.normal(0, 1, f) for x in xrange(n)]

def get_vectors(fn, n):
    dataset = []
    for line in gzip.open(fn):
        items = line.strip().split()
        dataset.append(numpy.array([float(x) for x in items[1:]]))
        if len(dataset) == n:
            break
    print len(dataset)
    return dataset

def get_avgs(dataset, n_iters=40):
    sums = numpy.zeros(3)
    for i in xrange(n_iters):
        k = random.choice(xrange(len(dataset)))
        dists = [distance.euclidean(dataset[j], dataset[k]) for j in xrange(len(dataset)) if j != k]
        dists.sort()
        sums += numpy.array([dists[0], dists[9], dists[-1]])
        print sums / (i+1)
    return sums / n_iters

fs_synt = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ts_synt = []
for f in fs_synt:
    print f, '...'
    dataset = get_gaussian(n, f)
    avgs = get_avgs(dataset)
    ts_synt.append(avgs)

fs_ts_real = []
for fn in sys.argv[1:]:
    dataset = get_vectors(fn, n)
    f = len(dataset[0])
    print f, '...'
    avgs = get_avgs(dataset)
    fs_ts_real.append((f, avgs))
fs_ts_real.sort()
fs_real = [f for f, t in fs_ts_real]
ts_real = [t for f, t in fs_ts_real]

def configure_ax():
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, 'both')
    ax.set_xlabel('Number of dimensions')
    ax.set_ylabel('Euclidean distance')
    return fig, ax

fig, ax = configure_ax()
ax.plot(fs_synt, ts_synt, 'x-', ms=10, mew=5)
ax.legend(['Distance to nearest neighbor',
           'Distance to neighbor #10',
           'Distance to furthest neighbor'], loc=4)
ax.set_title('%d points from a normal distribution' % n)
ax.annotate('Every point\'s neighborhood is the same!',
            xy=(fs_synt[-1], ts_synt[-1][0]),
            xytext=(fs_synt[-2], ts_synt[2][0]),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', va='bottom'
)
fig.savefig('knn_avg_dist_synt.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

ratio_synt = [ts_synt[i][2] / ts_synt[i][0] for i in xrange(len(ts_synt))]
ratio_real = [ts_real[i][2] / ts_real[i][0] for i in xrange(len(ts_real))]

fig, ax = configure_ax()
ax.plot(fs_synt, ratio_synt, 'x-', ms=10, mew=5, c='red')
ax.plot(fs_real, ratio_real, 'x-', ms=10, mew=5, c='blue')
ax.legend(['Synthetic data: furthest/closest distance ratio',
           'Real data: furthest/closest distance ratio'], loc=1)
ax.set_title('%d points from real word vectors' % n)
fig.savefig('knn_avg_dist_real_vs_synt.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
