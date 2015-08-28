import sys
import annoy
from matplotlib import pyplot as plt
import numpy

fn = sys.argv[1]
f = int(sys.argv[2])

a = annoy.AnnoyIndex(f, 'euclidean')
a.load(fn)

i = 0

def show_image(i):
    v = numpy.array(a.get_item_vector(i)).reshape((32, 32, 3))
    plt.imshow(v, interpolation='nearest')
    plt.show()

show_image(i)
for j in a.get_nns_by_item(i, 10, 10000):
    show_image(j)

    
