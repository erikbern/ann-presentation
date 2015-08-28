import shapely.geometry as sg
from shapely.ops import cascaded_union
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from voronoi import voronoi_polygons

fig, ax = plt.subplots()

def split_points(poly, points, voronoi, indices, lw=3.0, lo=0.0, hi=5.0/6.0, method='tree'):
    if len(indices) <= 1:
        x = [points[i][0] for i in indices]
        y = [points[i][1] for i in indices]
        c = hsv_to_rgb((lo+hi)/2, 1, 1)

        poly_vor = cascaded_union([sg.Polygon(voronoi[i]) for i in indices])

        if method == 'tree':
            pass
        elif method == 'voronoi':
            poly = poly_vor
        elif method == 'diff':
            poly = poly.symmetric_difference(poly_vor)
            c = 'red'

        if poly.geom_type == 'Polygon':
            polys = [poly]
        else:
            polys = poly.geoms

        for poly in polys:
            ax.add_patch(PolygonPatch(poly, fc=c, lw=0, zorder=0))
        plt.scatter(x, y, marker='x', zorder=99, c='black', s=1.0)

        return

    p1, p2 = [points[i] for i in random.sample(indices, 2)]
    v = p2-p1
    m = (p1+p2)/2
    a = np.dot(v, m)

    v_perp = np.array((v[1], -v[0]))

    big = 1e6
    halfplane_a = sg.Polygon(np.array([m+v_perp*big,  m+v*big, m-v_perp*big])).intersection(poly)
    halfplane_b = sg.Polygon(np.array([m+v_perp*big,  m-v*big, m-v_perp*big])).intersection(poly)

    ax.add_patch(PolygonPatch(halfplane_a, fc='none', lw=lw, zorder=1))
    ax.add_patch(PolygonPatch(halfplane_b, fc='none', lw=lw, zorder=1))

    indices_a = [i for i in indices if np.dot(points[i], v)-a > 0]
    indices_b = [i for i in indices if np.dot(points[i], v)-a < 0]

    split_points(halfplane_a, points, voronoi, indices_a, lw*0.8, lo, (lo+hi)/2, method)
    split_points(halfplane_b, points, voronoi, indices_b, lw*0.8, (lo+hi)/2, hi, method)

if __name__ == '__main__':
    points = np.random.randn(250, 2)
    voronoi = voronoi_polygons(points)

    inf = 1e9
    plane = sg.Polygon([(inf,inf), (inf,-inf), (-inf,-inf), (-inf,inf)])

    split_points(plane, points, voronoi, range(len(points)), method='voronoi')

    plt.axis('equal')
    plt.axis('off')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    # plt.show()
    plt.savefig('tree.png', dpi=1200, bbox_inches='tight', pad_inches=0, transparent=True)
