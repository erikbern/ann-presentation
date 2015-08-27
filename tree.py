import shapely.geometry as sg
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt


points = np.random.randn(1000, 2)

inf = 1e9
plane = sg.Polygon([(inf,inf), (inf,-inf), (-inf,-inf), (-inf,inf)])

fig, ax = plt.subplots()

def split_points(poly, points, lw=1.0):
    if len(points) < 10:
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        c = np.ones(len(points))*random.random()
        plt.scatter(x, y, c=c)

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

    split_points(halfplane_a, points_a, lw*0.8)
    split_points(halfplane_b, points_b, lw*0.8)

split_points(plane, points)

plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
