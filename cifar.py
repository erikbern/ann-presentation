import cPickle
import os
import annoy
import leargist
import PIL.Image

def read_cifar(d='cifar-10-batches-py'):
    for fn in os.listdir(d):
        fn = os.path.join(d, fn)
        with open(fn, 'rb') as f:
            if fn.find('_batch') == -1:
                continue

            data = cPickle.load(f)
            if 'data' not in data: continue

            for pixels, label in zip(data['data'], data['labels']):
                pixels = pixels.reshape((3, 32, 32)).transpose((1,2,0))
                yield pixels, label


def build(fn, f, fun): # lol @ parameters :)
    a = annoy.AnnoyIndex(f, 'euclidean')
    i = 0
    for pixels, label in read_cifar():
        a.add_item(i, fun(pixels))
        i += 1
        if i % 1000 == 0:
            print i, '...'

    a.build(100)
    a.save(fn)


def build_raw(fn):
    build(fn, 32*32*3, lambda pixels: pixels.reshape(32*32*3))


def build_leargist(fn):
    def fun(pixels):
        p = PIL.Image.fromarray(pixels)
        return leargist.color_gist(p)

    build(fn, 960, fun)


if __name__ == '__main__':
    if not os.path.exists('cifar_raw.annoy'):
        build_raw('cifar_raw.annoy')

    if not os.path.exists('cifar_leargist.annoy'):
        build_leargist('cifar_leargist.annoy')
