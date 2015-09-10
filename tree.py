import shapely.geometry as sg
from shapely.ops import cascaded_union
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from voronoi import voronoi_polygons
import functools
from sklearn.datasets import make_blobs

def split_points(ax, poly, points, voronoi, indices, lw=3.0, lo=0.0, hi=5.0/6.0, visitor=None, max_splits=99999, draw_splits=True, splits=None, seed='', leaf_size=1):
    random.seed(','.join([str(i) for i in indices]))

    if len(indices) <= leaf_size or max_splits == 0:
        x = [points[i][0] for i in indices]
        y = [points[i][1] for i in indices]
        c1 = hsv_to_rgb((lo+hi)/2, 1, 1)
        c2 = hsv_to_rgb(random.random()*5.0/6.0, 0.7+random.random()*0.3, 0.7+random.random()*0.3)

        poly_vor = cascaded_union([sg.Polygon(voronoi[i]) for i in indices])

        visitor.visit(ax, poly, poly_vor, c1, c2, x, y, splits)
        return

    random.seed(','.join([str(i) for i in indices]) + seed)
    p1, p2 = [points[i] for i in random.sample(indices, 2)]
    v = p2-p1
    m = (p1+p2)/2
    a = np.dot(v, m)

    v_perp = np.array((v[1], -v[0]))

    big = 1e6
    halfplane_a = sg.Polygon(np.array([m+v_perp*big,  m+v*big, m-v_perp*big])).intersection(poly)
    halfplane_b = sg.Polygon(np.array([m+v_perp*big,  m-v*big, m-v_perp*big])).intersection(poly)

    if draw_splits:
        ax.add_patch(PolygonPatch(halfplane_a, fc='none', lw=lw, zorder=1))
        ax.add_patch(PolygonPatch(halfplane_b, fc='none', lw=lw, zorder=1))

    indices_a = [i for i in indices if np.dot(points[i], v)-a > 0]
    indices_b = [i for i in indices if np.dot(points[i], v)-a < 0]

    split_points(ax, halfplane_a, points, voronoi, indices_a, lw*0.8, lo, (lo+hi)/2, visitor, max_splits-1, draw_splits, (splits, v, a), seed, leaf_size)
    split_points(ax, halfplane_b, points, voronoi, indices_b, lw*0.8, (lo+hi)/2, hi, visitor, max_splits-1, draw_splits, (splits, -v, -a), seed, leaf_size)

def draw_poly(ax, poly, c, lw=0):
    if poly.geom_type == 'Polygon':
        polys = [poly]
    else:
        polys = poly.geoms

    for poly in polys:
        ax.add_patch(PolygonPatch(poly, fc=c, lw=lw, zorder=0))

def scatter(ax, x, y):
    ax.scatter(x, y, marker='x', zorder=99, c='black', s=10.0)

class Visitor(object):
    def visit(self, ax, poly, c1, c2, x, y, splits):
        pass

class TreeVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        draw_poly(ax, poly, c1)
        scatter(ax, x, y)

class VoroVisitor(Visitor):
    def __init__(self, randomize_c=False):
        self._randomize_c = randomize_c

    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        if self._randomize_c:
            c = c2
        else:
            c = c1
        draw_poly(ax, poly_vor, c, lw=0.2)
        scatter(ax, x, y)

class ScatterVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        scatter(ax, x, y)

class HeapVisitor(Visitor):
    def __init__(self, p, alpha=1.0):
        self._p = p
        self._alpha = alpha

    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        margin = float('inf')
        while splits:
            splits, v, a = splits
            margin = min(margin, np.dot(self._p, v) - a)

        c = plt.get_cmap('YlOrRd')(1 + margin * 0.5)
        c = (c[0], c[1], c[2], self._alpha)

        draw_poly(ax, poly, c)
        ax.scatter(self._p[0], self._p[1], marker='x', zorder=99, c='red', s=100.0)
        scatter(ax, x, y)

class ForestVisitor(Visitor):
    def __init__(self, alpha=1.0):
        self._alpha = alpha

    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        draw_poly(ax, poly, c2 + (self._alpha,))
        draw_poly(ax, poly_vor, c=(0, 0, 0, 0), lw=0.2)
        scatter(ax, x, y)

def get_points():
    np.random.seed(0)
    X, y = make_blobs(500, 2, centers=10, center_box=(-4, 4))
    return X

def main():
    points = get_points()
    voronoi = voronoi_polygons(points)

    inf = 1e9
    plane = sg.Polygon([(inf,inf), (inf,-inf), (-inf,-inf), (-inf,inf)])

    p = np.random.randn(2)
    plots = [('scatter', ScatterVisitor(), 999, False, 1, 1),
             ('voronoi', VoroVisitor(True), 999, False, 1, 1),
             ('tree-1', TreeVisitor(), 1, True, 1, 1),
             ('tree-2', TreeVisitor(), 2, True, 1, 1),
             ('tree-3', TreeVisitor(), 3, True, 1, 1),
             ('tree-full', TreeVisitor(), 999, True, 1, 1),
             ('tree-full-K', TreeVisitor(), 999, True, 1, 10),
             ('voronoi-tree-1', VoroVisitor(), 1, True, 1, 1),
             ('voronoi-tree-2', VoroVisitor(), 2, True, 1, 1),
             ('voronoi-tree-3', VoroVisitor(), 3, True, 1, 1),
             ('heap', HeapVisitor(p), 999, True, 1, 1),
             ('forest', ForestVisitor(0.05), 999, False, 40, 1),
             ('forest-heap', HeapVisitor(p, 0.05), 999, False, 40, 1)]

    for tag, visitor, max_splits, draw_splits, n_iterations, leaf_size in plots:
        fn = tag + '.png'
        print fn, '...'

        fig, ax = plt.subplots()

        for iteration in xrange(n_iterations):
            print iteration, '...'
            split_points(ax, plane, points, voronoi, range(len(points)), visitor=visitor, max_splits=max_splits, draw_splits=draw_splits, seed=(iteration > 1 and str(iteration) or ''), leaf_size=leaf_size)

        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

if __name__ == '__main__':
    main()
