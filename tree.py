import shapely.geometry as sg
from shapely.ops import cascaded_union
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
try:
    from voronoi import voronoi_polygons
except:
    def voronoi_polygons(x): pass
import functools
from sklearn.datasets import make_blobs
import pygraphviz
from scipy.spatial import distance

def split_points(ax, graph, poly, points, voronoi, indices, lw=3.0, lo=0.0, hi=5.0/6.0, visitor=None, max_splits=99999, draw_splits=True, splits=None, seed='', leaf_size=1, parent_node_id=None):
    indices_str = ','.join([str(i) for i in indices])
    random.seed(indices_str)
    node_id = hash(indices_str + seed)

    leaf = (len(indices) <= leaf_size or max_splits == 0)

    visitor.draw_node(graph, node_id, leaf, indices, lo, hi, splits)
    if parent_node_id:
        graph.add_edge(parent_node_id, node_id)

    if leaf:
        x = [points[i][0] for i in indices]
        y = [points[i][1] for i in indices]
        c1 = hsv_to_rgb((lo+hi)/2, 1, 1)
        c2 = hsv_to_rgb(random.random()*5.0/6.0, 0.7+random.random()*0.3, 0.7+random.random()*0.3)
        poly_vor = None
        if voronoi:
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

    if max_splits == 1:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', lw=2.0, zorder=2)

    indices_a = [i for i in indices if np.dot(points[i], v)-a > 0]
    indices_b = [i for i in indices if np.dot(points[i], v)-a < 0]

    split_points(ax, graph, halfplane_a, points, voronoi, indices_a, lw*0.8, lo, (lo+hi)/2, visitor, max_splits-1, draw_splits, (splits, v, a), seed, leaf_size, node_id)
    split_points(ax, graph, halfplane_b, points, voronoi, indices_b, lw*0.8, (lo+hi)/2, hi, visitor, max_splits-1, draw_splits, (splits, -v, -a), seed, leaf_size, node_id)

def draw_poly(ax, poly, c, **kwargs):
    if poly.geom_type == 'Polygon':
        polys = [poly]
    else:
        polys = poly.geoms

    for poly in polys:
        ax.add_patch(PolygonPatch(poly, fc=c, zorder=0, **kwargs))

def scatter(ax, x, y):
    ax.scatter(x, y, marker='x', zorder=99, c='black', s=10.0)

class Visitor(object):
    def visit(self, ax, poly, c1, c2, x, y, splits):
        pass

    def node_attrs(self, node_id, leaf, indices, lo, hi):
        label = leaf and len(indices) or ''
        shape = leaf and 'circle' or 'square'
        return dict(label=label, style='filled', fillcolor='%f 1.0 1.0' % ((lo+hi)/2), fontsize=24, fontname='bold', shape=shape)

    def draw_node(self, graph, node_id, leaf, indices, lo, hi, splits):
        graph.add_node(node_id, **self.node_attrs(node_id, leaf, indices, lo, hi))

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

class ScatterNNsVisitor(Visitor):
    def __init__(self, p, nns=20):
        self._p = p
        self._nns = nns

    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        scatter(ax, x, y)
        ax.plot(self._p[0], self._p[1], marker='x', zorder=99, c='blue', ms=3, mew=1)
        xys = [np.array((x, y)) for x, y in zip(x, y)]
        xys.sort(key=lambda p: distance.euclidean(self._p, p))
        for p in xys[:self._nns]:
            ax.plot([self._p[0], p[0]], [self._p[1], p[1]], 'r-')

        c = plt.Circle(self._p, distance.euclidean(self._p, xys[self._nns-1]), edgecolor='red', zorder=99, lw=1.0, fill=False, linestyle='dashed')
        ax.add_artist(c)

class HeapVisitor(Visitor):
    def __init__(self, p, min_margin=-0.5):
        self._p = p
        self._min_margin = min_margin

    def get_margin(self, splits):
        margin = float('inf')
        while splits:
            splits, v, a = splits
            margin = min(margin, np.dot(self._p, v) - a)

        return margin
    
    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        margin = self.get_margin(splits)
        c = (margin > self._min_margin and c1 or 'none')

        draw_poly(ax, poly, c)
        ax.plot(self._p[0], self._p[1], marker='x', zorder=99, c='red', ms=10, mew=5)
        scatter(ax, x, y)

    def draw_node(self, graph, node_id, leaf, indices, lo, hi, splits):
        attrs = self.node_attrs(node_id, leaf, indices, lo, hi)
        margin = self.get_margin(splits)
        if margin >= self._min_margin:
            attrs['penwidth'] = 10.0
        else:
            attrs['fillcolor'] = 'white'
        graph.add_node(node_id, **attrs)

class NNsVisitor(Visitor):
    def __init__(self, p, dist=1.0):
        self._p = p
        self._dist = dist

    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        draw_poly(ax, poly, c1)
        scatter(ax, x, y)
        c = plt.Circle(self._p, self._dist, edgecolor='white', zorder=99, lw=2.0, fill=False)
        ax.add_artist(c)
        ax.plot(self._p[0], self._p[1], marker='x', zorder=99, c='white', ms=10, mew=5)

def get_points():
    np.random.seed(0)
    X, y = make_blobs(500, 2, centers=10, center_box=(-4, 4))
    return X

def main():
    points = get_points()
    voronoi = voronoi_polygons(points)

    inf = 1e9
    plane = sg.Polygon([(inf,inf), (inf,-inf), (-inf,-inf), (-inf,inf)])

    p = np.array([0, 0]) # np.random.randn(2)
    q = np.array([-2, -2])
    plots = [('scatter', ScatterVisitor(), 0, False, '', 10),
             ('scatter-nns-5', ScatterNNsVisitor(q, 5), 0, False, '', 10),
             ('scatter-nns-20', ScatterNNsVisitor(q, 20), 0, False, '', 10),
             ('scatter-nns-100', ScatterNNsVisitor(q, 100), 0, False, '', 10),
             ('voronoi', VoroVisitor(True), 999, False, '', 1),
             ('tree-1', TreeVisitor(), 1, True, '', 10),
             ('tree-2', TreeVisitor(), 2, True, '', 10),
             ('tree-3', TreeVisitor(), 3, True, '', 10),
             ('tree-full', TreeVisitor(), 999, True, '', 10),
             ('tree-full-K', TreeVisitor(), 999, True, '', 10),
             ('tree-point', NNsVisitor(p), 99, True, '', 10),
             ('voronoi-tree-1', VoroVisitor(), 1, True, '', 10),
             ('voronoi-tree-2', VoroVisitor(), 2, True, '', 10),
             ('voronoi-tree-3', VoroVisitor(), 3, True, '', 10),
             ('heap', HeapVisitor(p), 999, True, '', 10),
             ('heap-pos', HeapVisitor(p, 0), 999, True, '', 10),
             ('heap-1', HeapVisitor(p), 999, True, '1', 10),
             ('heap-2', HeapVisitor(p), 999, True, '2', 10),
             ('heap-3', HeapVisitor(p), 999, True, '3', 10),
             ('heap-4', HeapVisitor(p), 999, True, '4', 10),
             ('heap-5', HeapVisitor(p), 999, True, '5', 10)]

    for tag, visitor, max_splits, draw_splits, seed, leaf_size in plots:
        fn = tag + '.png'
        print fn, '...'

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        graph = pygraphviz.AGraph()
        split_points(ax, graph, plane, points, voronoi, range(len(points)), visitor=visitor, max_splits=max_splits, draw_splits=draw_splits, seed=seed, leaf_size=leaf_size)

        ax.set_xlim(-8, 8)
        ax.set_ylim(-6, 6)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        fig.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

        graph.layout(prog='dot')
        graph.draw(tag + '-graphviz.png')


if __name__ == '__main__':
    main()
