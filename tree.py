import shapely.geometry as sg
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

points = np.random.randn(10000, 2)

inf = 1e9
plane = sg.Polygon([(inf,inf), (inf,-inf), (-inf,-inf), (-inf,inf)])

fig, ax = plt.subplots()

def split_points(poly, points, lw=1.0, lo=0.0, hi=2.0/3.0):
    if len(points) < 10:
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        c = hsv_to_rgb((lo+hi)/2, 1, 1)

        c = np.array([c] * len(points))
        plt.scatter(x, y, c=c, marker='x')

        # ax.add_patch(PolygonPatch(poly, fc=c, lw=0, alpha=0.2))

        return
    
    p1, p2 = random.sample(points, 2)
    v = p2-p1
    m = (p1+p2)/2
    a = np.dot(v, m)

    v_perp = np.array((v[1], -v[0]))

    big = 1e6
    halfplane_a = sg.Polygon(np.array([m+v_perp*big,  m+v*big, m-v_perp*big])).intersection(poly)
    halfplane_b = sg.Polygon(np.array([m+v_perp*big,  m-v*big, m-v_perp*big])).intersection(poly)

    ax.add_patch(PolygonPatch(halfplane_a, fc='none', lw=lw))
    ax.add_patch(PolygonPatch(halfplane_b, fc='none', lw=lw))

    points_a = [p for p in points if np.dot(p, v)-a > 0]
    points_b = [p for p in points if np.dot(p, v)-a < 0]

    split_points(halfplane_a, points_a, lw*0.8, lo, (lo+hi)/2)
    split_points(halfplane_b, points_b, lw*0.8, (lo+hi)/2, hi)

split_points(plane, points)

plt.axis('equal')
plt.axis('off')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
# plt.show()
plt.savefig('tree.png', dpi=1200, bbox_inches='tight', pad_inches=0, transparent=True)
