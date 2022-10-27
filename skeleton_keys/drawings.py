import shapely
import shapely.ops
import numpy as np
from functools import partial
from neuron_morphology.snap_polygons.geometries import (
    Geometries, make_scale, clear_overlaps, closest_from_stack,
    get_snapped_polys, find_vertical_surfaces, select_largest_subpolygon
)
from neuron_morphology.transforms.geometry import get_vertices_from_two_lines
from shapely.geometry.polygon import Polygon, LineString, orient


def snap_hand_drawn_polygons(
    layer_polygons,
    pia_surface,
    wm_surface,
    layer_order,
    working_scale=1.0 / 4, # is this right? default from InputParameters
):
    """ Snap together hand-drawn polygons """
    pia_wm_vertices = get_vertices_from_two_lines(pia_surface["path"],
                                                  wm_surface["path"])
    bounds = Polygon(pia_wm_vertices)

    geometries = Geometries()
    geometries.register_polygons(layer_polygons)
    geometries.register_surface("pia", pia_surface["path"])
    geometries.register_surface("wm", wm_surface["path"])

    scale_transform = make_scale(working_scale)
    working_geo = geometries.transform(scale_transform)

    raster_stack = working_geo.rasterize()
    clear_overlaps(raster_stack)
    closest, closest_names = closest_from_stack(raster_stack)
    multipolygon_resolver = partial(
        select_largest_subpolygon,
        error_threshold=-float("inf")
    )
    snapped_polys = get_snapped_polys(closest, closest_names, multipolygon_resolver)

    result_geos = Geometries()
    result_geos.register_polygons(snapped_polys)

    result_geos = (result_geos
        .transform(
            lambda hr, vt: (
                hr + working_geo.close_bounds.horigin,
                vt + working_geo.close_bounds.vorigin
            )
        )
        .transform(make_scale(1.0 / working_scale))
    )

    for key in list(result_geos.polygons.keys()):
        result_geos.polygons[key] = result_geos.polygons[key].intersection(bounds)
        # keep largest polygon if multiple geometries
#         print(key, result_geos.polygons[key].geom_type)
        if result_geos.polygons[key].geom_type in ["MultiPolygon", "GeometryCollection"]:
            poly_area = 0
            for p in result_geos.polygons[key]:
                if p.geom_type == "Polygon" and p.area > poly_area:
                    keep_poly = p
                    poly_area = p.area
            result_geos.polygons[key] = keep_poly


    boundaries = find_vertical_surfaces(
        result_geos.polygons,
        layer_order,
        pia=geometries.surfaces["pia"],
        white_matter=geometries.surfaces["wm"]
    )

    result_geos.register_surfaces(boundaries)
    result_geos.register_surface("pia", pia_surface["path"])
    result_geos.register_surface("wm", wm_surface["path"])

    results = result_geos.to_json()
    return results


def convert_and_translate_snapped_to_microns(polys_surfs, res, translation):
    """ Convert surfaces and layers from pixels to microns and translate to specified location"""
    surfs = []
    for s in polys_surfs["surfaces"]:
        path_coords = np.asarray(s["path"])
        path_coords *= res
        path_coords = path_coords + translation
        s["path"] = path_coords.tolist()
        surfs.append(s)

    polys = []
    for p in polys_surfs["polygons"]:
        path_coords = np.asarray(p["path"])
        path_coords *= res
        path_coords = path_coords + translation
        p["path"] = path_coords.tolist()
        polys.append(p)

    return {"surfaces": surfs, "polygons": polys}


def perimeter_of_layers(boundaries):
    """ Get overall perimeter of a set of layer boundaries

    Parameters
    ----------
    boundaries : dict
        Dictionary with layer names as keys and boundary coordinates as values

    Returns
    -------
        perimeter : Polygon
            Shapely polygon of the perimeter
    """

    polys = [Polygon(b) for b in boundaries.values()]
    perimeter = shapely.ops.unary_union(polys)
    return perimeter


def simplify_and_find_corners(perimeter, tolerance=2, n_corners=4):
    """ Simplify a perimeter boundary and identify corners

    Corners are the top `n_corners` vertices with angles closest to
    90 degrees.

    Parameters
    ----------
    perimeter : Polygon
        Shapely polygon of the perimeter
    tolerance : float, default 2
        Tolerance parameter for shapely's Polygon.simplify()
    n_corners : int, default 4
        Number of corners to identify

    Returns
    -------
    simplified_perimeter : Polygon
        Perimeter with fewer vertices
    corners : (n_corners, ) array
        Indexes of corner vertices
    """

    # Simplify perimeter
    simple_perim = orient(perimeter.simplify(tolerance=tolerance))

    # Try to find corners
    corner_inds = _assign_corners(simple_perim)

    return simple_perim, corner_inds


def angles_around_polygon(poly):
    """ Calculate vertex angles for polygon exterior"""
    coords = np.array(poly.exterior.coords)

    ab_vecs = coords[:-1, :] - coords[1:, :]
    ca_vecs = np.vstack([coords[2:, :], coords[1, :]]) - coords[1:, :]

    angles = np.arctan2(np.cross(ab_vecs, ca_vecs), np.einsum('ij,ij->i', ab_vecs, ca_vecs))

    return np.hstack([angles[-1], angles[:-1]])


def _assign_corners(perimeter):
    perim_coords = np.array(perimeter.exterior.coords)
    vertex_angles = angles_around_polygon(perimeter)
    enclosing_rect = perimeter.minimum_rotated_rectangle

    rect_coords = np.array(enclosing_rect.exterior.coords)

    angle_diff = np.abs(vertex_angles + np.pi / 2)
    sorted_angle_inds = np.argsort(angle_diff)
    keep_going = True
    corners_guess = np.sort(sorted_angle_inds[:4])
    next_guess_ind = 4
    best_score = np.inf
    full_dists_to_rect = distance.cdist(perim_coords, rect_coords)
    while(keep_going):
        total_dist = distances_for_guess(full_dists_to_rect, corners_guess)
        print("-------")
        print(corners_guess)
        print(total_dist, np.sum(total_dist))

        # try next point - sub for each
        best_guess_set = None
        best_guess_score = np.inf
        for i in range(4):
            new_guess = corners_guess.copy()
            new_guess[i] = sorted_angle_inds[next_guess_ind]
            new_guess = np.sort(new_guess)
            print("new try", new_guess)
            new_dist = distances_for_guess(full_dists_to_rect, new_guess)
            print(new_dist, np.sum(new_dist))
            if np.sum(new_dist) < best_guess_score:
                best_guess_score = np.sum(new_dist)
                best_guess_set = new_guess

        if best_guess_score < np.sum(total_dist):
            keep_going = True
            next_guess_ind += 1
            corners_guess = best_guess_set
        else:
            keep_going = False

    return corners_guess

def _distances_for_corner_guess(dist_mat, corners_guess):
    distances_to_rect = dist_mat[corners_guess, :]

    # Find corner closest to rectangle corner
    closest_rect = np.argmin(np.min(distances_to_rect, axis=0))
    closest_corner_ind = np.argmin(distances_to_rect[:, closest_rect])

    # Assign corners in order starting from closest
    assignments = [(corners_guess[i % 4], r % 4)
                   for i, r in zip(range(closest_corner_ind, closest_corner_ind + 4),
                                   range(closest_rect, closest_rect + 4))]
    total_dist = []
    for a in assignments:
        total_dist.append(dist_mat[a[0], a[1]])
    return total_dist

def identify_pia_and_wm_sides(perimeter, corners, boundaries):
    """ Find pia and wm sides

    Parameters
    ----------
    simplified_perimeter : Polygon
        Perimeter with fewer vertices
    corners : (n_corners, ) array
        Indexes of corner vertices
    boundaries : dict
        Dictionary with layer names as keys and boundary coordinates as values

    Returns
    -------
    pia_side : LineString
        Pia side of cortical boundaries
    wm_side : LineString
        Wm side of cortical boundaries
    """

    layer_order = [
        "Isocortex layer 1",
        "Isocortex layer 2/3",
        "Isocortex layer 4",
        "Isocortex layer 5",
        "Isocortex layer 6a",
        "Isocortex layer 6b",
    ]
    layer_polys = [Polygon(b) for b in boundaries.values()]

    # Use corners to define sides
    n_sides = len(corners)
    if n_sides % 2 != 0:
        raise ValueError("Number of corners is not even")
    half_n_sides = n_sides // 2
    side_pairs = [(i, i + half_n_sides) for i in range(half_n_sides)]

    sides = []
    coords = np.array(perimeter.exterior.coords)
    for i in range(len(corners)):
        ind_start = corners[i]
        ind_end = corners[(i + 1) % n_sides]
        if ind_start < ind_end:
            sides.append(LineString(coords[ind_start:ind_end + 1, :]))
        else:
            combo_coords = np.vstack([coords[ind_start:-1], coords[:ind_end + 1]])
            sides.append(LineString(combo_coords))

    # Find sides that represent top/bottom
    bigger_dist_to_pair = {}
    for pair in side_pairs:
        bigger_dist_to_pair[pair] = []
        for p in layer_polys:
            bigger_dist_to_pair[pair].append(max(p.distance(sides[pair[0]]), p.distance(sides[pair[1]])))

    max_avg_dist = -np.inf
    top_bottom_pair = None
    for pair in side_pairs:
        if np.mean(bigger_dist_to_pair[pair]) > max_avg_dist:
            top_bottom_pair = pair
            max_avg_dist = np.mean(bigger_dist_to_pair[pair])

    # Determine top (pia) side
    for l in layer_order:
        if l in boundaries:
            top_poly = Polygon(boundaries[l])
            break

    if top_poly.distance(sides[top_bottom_pair[0]]) < top_poly.distance(sides[top_bottom_pair[1]]):
        pia_side = sides[top_bottom_pair[0]]
        wm_side = sides[top_bottom_pair[1]]
    else:
        pia_side = sides[top_bottom_pair[1]]
        wm_side = sides[top_bottom_pair[0]]

    return pia_side, wm_side

def simplify_layer_boundaries(boundaries, tolerance):
    """ Simplify layer boundary drawings

    Parameters
    ----------
    boundaries : dict
        Dictionary with layer names as keys and boundary coordinates as values
    tolerance : float
        Tolerance parameter for shapely's Polygon.simplify()

    Returns
    -------
    simplified_boundaries : dict
        Dictionary with layer names as keys and simplified boundary coordinates as values
    """
    simplified_boundaries = {}
    for k, v in boundaries.items():
        b_poly = Polygon(v).simplify(tolerance=tolerance)
        simplified_boundaries[k] = np.array(b_poly.exterior.coords)
    return simplified_boundaries