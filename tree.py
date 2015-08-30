import shapely.geometry as sg
from shapely.ops import cascaded_union
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from voronoi import voronoi_polygons
import functools

def split_points(ax, poly, points, voronoi, indices, lw=3.0, lo=0.0, hi=5.0/6.0, visitor=None, max_splits=99999, draw_splits=True, splits=None):
    random.seed(','.join([str(i) for i in indices]))

    if len(indices) <= 1 or max_splits == 0:
        x = [points[i][0] for i in indices]
        y = [points[i][1] for i in indices]
        # c = hsv_to_rgb((lo+hi)/2, 1, 1)
        c = hsv_to_rgb(random.random()*5.0/6.0, 0.7+random.random()*0.3, 0.7+random.random()*0.3)

        poly_vor = cascaded_union([sg.Polygon(voronoi[i]) for i in indices])

        visitor.visit(ax, poly, poly_vor, c, x, y, splits)
        return

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

    split_points(ax, halfplane_a, points, voronoi, indices_a, lw*0.8, lo, (lo+hi)/2, visitor, max_splits-1, draw_splits, (splits, v, a))
    split_points(ax, halfplane_b, points, voronoi, indices_b, lw*0.8, (lo+hi)/2, hi, visitor, max_splits-1, draw_splits, (splits, -v, -a))

def draw_poly(ax, poly, c, lw=0):
    if poly.geom_type == 'Polygon':
        polys = [poly]
    else:
        polys = poly.geoms

    for poly in polys:
        ax.add_patch(PolygonPatch(poly, fc=c, lw=lw, zorder=0))

def scatter(ax, x, y):
    plt.scatter(x, y, marker='x', zorder=99, c='black', s=10.0)

class Visitor(object):
    def visit(self, ax, poly, c, x, y, splits):
        pass

class TreeVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c, x, y, splits):
        draw_poly(ax, poly, c)
        scatter(ax, x, y)

class VoroVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c, x, y, splits):
        draw_poly(ax, poly_vor, c, lw=0.2)
        scatter(ax, x, y)

class ScatterVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c, x, y, splits):
        scatter(ax, x, y)

class HeapVisitor(Visitor):
    def __init__(self, p):
        print p
        self._p = p

    def visit(self, ax, poly, poly_vor, c, x, y, splits):
        margin = float('inf')
        while splits:
            splits, v, a = splits
            margin = min(margin, np.dot(self._p, v) - a)

        c = plt.get_cmap('YlOrRd')(1 + margin * 0.5)
        draw_poly(ax, poly, c)
        plt.scatter(self._p[0], self._p[1], marker='x', zorder=99, c='red', s=100.0)


def get_points():
    np.random.seed(4)
    return np.random.randn(200, 2)

def main():
    points = get_points()
    voronoi = voronoi_polygons(points)

    inf = 1e9
    plane = sg.Polygon([(inf,inf), (inf,-inf), (-inf,-inf), (-inf,inf)])

    plots = [('none', ScatterVisitor(), 999, False),
             ('voronoi', VoroVisitor(), 999, False),
             ('tree', TreeVisitor(), 1, True),
             ('tree', TreeVisitor(), 2, True),
             ('tree', TreeVisitor(), 3, True),
             ('tree', TreeVisitor(), 999, True),
             ('voronoi', VoroVisitor(), 1, True),
             ('voronoi', VoroVisitor(), 2, True),
             ('voronoi', VoroVisitor(), 3, True),
             ('heap', HeapVisitor(np.random.randn(2)), 999, True)]

    for tag, visitor, max_splits, draw_splits in plots:
        fn = 'tree-%s-%d-%s.png' % (tag, max_splits, draw_splits)
        print fn, '...'

        fig, ax = plt.subplots()

        split_points(ax, plane, points, voronoi, range(len(points)), visitor=visitor, max_splits=max_splits, draw_splits=draw_splits)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

if __name__ == '__main__':
    main()
