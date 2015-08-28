import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from colorsys import hsv_to_rgb
import random

# Code stolen mostly from https://gist.github.com/pv/8036995

def voronoi_finite_polygons_2d(points, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    # compute Voronoi tesselation
    vor = Voronoi(points)

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    new_vertices = np.asarray(new_vertices)
    polygons = []
    for new_region in new_regions:
        polygons.append(new_vertices[new_region])
        
    return polygons

if __name__ == '__main__':
    # make up data points
    np.random.seed(1234)
    points = np.random.randn(1000, 2)

    # plot
    polygons = voronoi_finite_polygons_2d(points)
    print polygons

    # colorize
    for polygon in polygons:
        r = random.random()*5.0/6.0
        c = hsv_to_rgb(r, 1, 1)
        plt.fill(*zip(*polygon), fc=c, zorder=0)

    plt.scatter(points[:,0], points[:,1], marker='x', s=1.0, zorder=2, c='black')

    plt.axis('equal')
    plt.axis('off')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    # plt.show()
    plt.savefig('voro.png', dpi=1200, bbox_inches='tight', pad_inches=0, transparent=True)
