import os
from urllib import urlretrieve
import annoy
import random
import PIL.Image, PIL.ImageOps
import numpy
import util

annoy_fn = 'mnist.annoy'
data_fn = 'mnist.pkl.gz'

if not os.path.exists(annoy_fn):
    if not os.path.exists(data_fn):
        print 'downloading'
        urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', data_fn)

    a = annoy.AnnoyIndex(784, 'euclidean')
    for i, pic in util.get_vectors(data_fn):
        a.add_item(i, pic)

    print 'building'
    a.build(10)
    a.save(annoy_fn)

a = annoy.AnnoyIndex(784, 'euclidean')
a.load(annoy_fn)

pics = 5
nns = 10
img_size = 100
margin = 16

main_image = PIL.Image.new('RGB', (img_size * nns + margin, img_size * pics), 'white')

for pic in xrange(pics):
    i = random.randint(0, a.get_n_items() - 1)
    for index, j in enumerate(a.get_nns_by_item(i, 10, 1000)):
        v = a.get_item_vector(j)
        w = (numpy.array(v)*255).astype(numpy.uint8).reshape(28, 28)
        image = PIL.Image.fromarray(w)
        image = PIL.ImageOps.fit(image, (img_size, img_size)) # , PIL.Image.ANTIALIAS)
        if index == 0:
            image.save('seed.jpg')
        
        main_image.paste(image, (index * img_size + margin * int(index > 0), pic * img_size))

main_image.save('mnist_strips.jpg')



