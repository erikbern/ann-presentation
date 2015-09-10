import os
from urllib import urlretrieve
import gzip
import cPickle
import annoy
import random
import PIL.Image, PIL.ImageOps
import numpy

annoy_fn = 'mnist.annoy'
data_fn = 'mnist.pkl.gz'

if not os.path.exists(annoy_fn):
    if not os.path.exists(data_fn):
        print 'downloading'
        urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', data_fn)

    a = annoy.AnnoyIndex(784, 'euclidean')
    i = 0
    f = gzip.open(data_fn)
    for pics, labels in cPickle.load(f):
        for pic in pics:
            a.add_item(i, pic)
            i += 1

    print 'building'
    a.build(10)
    a.save(annoy_fn)

a = annoy.AnnoyIndex(784, 'euclidean')
a.load(annoy_fn)

nns = 10
img_size = 100
margin = 16

main_image = PIL.Image.new('RGB', (img_size * nns + margin, img_size), 'white')

i = random.randint(0, a.get_n_items() - 1)
for index, j in enumerate(a.get_nns_by_item(i, 10, 1000)):
    v = a.get_item_vector(j)
    v = (numpy.array(v)*255).astype(numpy.uint8).reshape(28, 28)
    image = PIL.Image.fromarray(v)
    image = PIL.ImageOps.fit(image, (img_size, img_size)) # , PIL.Image.ANTIALIAS)
    main_image.paste(image, (index * img_size + margin * int(index > 0), 0))

main_image.save('mnist_strip.jpg')



