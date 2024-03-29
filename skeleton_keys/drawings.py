import shapely
import shapely.ops
import numpy as np
import scipy.spatial.distance as distance
from functools import partial
from neuron_morphology.snap_polygons.geometries import (
    Geometries, make_scale, clear_overlaps, closest_from_stack,
    get_snapped_polys, find_vertical_surfaces, select_largest_subpolygon
)
from neuron_morphology.transforms.geometry import get_vertices_from_two_lines
from shapely.geometry.polygon import Point, Polygon, LineString, orient


def snap_hand_drawn_polygons(
        layer_polygons,
        pia_surface,
        wm_surface,
        layer_order,
        working_scale=1.0 / 4,  # is this right? default from InputParameters
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


def remove_duplicate_coordinates_from_drawings(drawings):
    """
    Will remove any duplicate coordinates found in a drawing dictionary.

    :param drawings: List or Dict of polygon/surface drawings
    :return: List or Dict of polygon/surface drawings having unique x-y coordinates with order preserved
    """
    if isinstance(drawings, list):
        cleaned_drawings = []
        for poly_dict in drawings:
            path_arr = np.array(poly_dict['path'])
            _, idx = np.unique(path_arr, axis=0, return_index=True)
            poly_dict['path'] = path_arr[sorted(idx)].tolist()

            cleaned_drawings.append(poly_dict)

        return cleaned_drawings

    elif isinstance(drawings, dict):
        path_arr = np.array(drawings['path'])
        _, idx = np.unique(path_arr, axis=0, return_index=True)
        drawings['path'] = path_arr[sorted(idx)].tolist()

        return drawings
    else:
        raise TypeError("Must provide either list of drawings, or individual drawing dictionary")


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

    polys = [Polygon(b) for b in boundaries.values() if np.size(b) > 0]
    perimeter = shapely.ops.unary_union(polys)
    return perimeter


def simplify_and_find_sides(perimeter, boundaries, tolerance=2):
    """ Simplify a perimeter boundary and identify corners

    Corners are the top `n_corners` vertices with angles closest to
    90 degrees.

    Parameters
    ----------
    perimeter : Polygon
        Shapely polygon of the perimeter
    tolerance : float, default 2
        Tolerance parameter for shapely's Polygon.simplify()

    Returns
    -------
    simplified_perimeter : Polygon
        Perimeter with fewer vertices
    corners : (n_corners, ) array
        Indexes of corner vertices
    """

    # Simplify perimeter
    simple_perim = orient(perimeter.simplify(tolerance=tolerance))

    # Get polygons for layers
    layer_polys = {}
    for k, v in boundaries.items():
        layer_polys[k] = Polygon(v)

    # Get an initial set of corners to optimize from
    init_guess = _find_initial_corners(simple_perim, boundaries)

    # Break apart into sides and get some initial metrics
    sides, corner_sets = _split_poly_into_sides(simple_perim, init_guess)

    init_touches = _count_layer_touches(sides, layer_polys)
    init_tort = _tortuosity_of_sides(sides)


    # Guess at the best candidate for a "vertical side" (i.e. not pia or wm)
    best_vert_side_init = _initial_vertical_side_guess(init_touches, init_tort)

    # Optimize the opposite side first
    opp_vert_side_init = (best_vert_side_init + 2) % 4

    # Find the worse problem corner
    # pia/wm should minimize the number of layers they touch, so find the
    # one that touches more

    adj_a = (opp_vert_side_init - 1) % 4
    adj_b = (opp_vert_side_init + 1) % 4
    if init_touches[adj_a] < init_touches[adj_b]:
        tmp = adj_a
        adj_a = adj_b
        adj_b = tmp

    # Optimize opp / adj_a
    new_corners, new_opp_vert_set = _optimize_corner_location(
        init_guess, sides, corner_sets,
        opp_vert_side_init, adj_a,
        simple_perim, layer_polys)
    sides, corner_sets = _reset_sides(simple_perim, new_corners,
        opp_vert_side_init, new_opp_vert_set)

    # Optimize opp / adj_b
    new_corners, new_opp_vert_set = _optimize_corner_location(
        new_corners, sides, corner_sets,
        opp_vert_side_init, adj_b,
        simple_perim, layer_polys)
    sides, corner_sets = _reset_sides(simple_perim, new_corners,
        opp_vert_side_init, new_opp_vert_set)

    # Optimize initial best / adj_a
    new_corners, new_best_vert_set = _optimize_corner_location(
        new_corners, sides, corner_sets,
        best_vert_side_init, adj_a,
        simple_perim, layer_polys)
    sides, corner_sets = _reset_sides(simple_perim, new_corners,
        best_vert_side_init, new_best_vert_set)

    # Optimize initial best / adj_b
    new_corners, new_best_vert_set = _optimize_corner_location(
        new_corners, sides, corner_sets,
        best_vert_side_init, adj_b,
        simple_perim, layer_polys)
    sides, corner_sets = _reset_sides(simple_perim, new_corners,
        best_vert_side_init, new_best_vert_set)

    # Refine the top / bottom edges (adj_a & adj_b)
    adj_a_corners = corner_sets[adj_a]
    adj_b_corners = corner_sets[adj_b]

    adj_a_corners = _expand_to_touch_additional_layer(
        adj_a_corners, simple_perim, layer_polys)
    adj_b_corners = _expand_to_touch_additional_layer(
        adj_b_corners, simple_perim, layer_polys)

    adj_a_corners = _trim_edges(
        adj_a_corners, simple_perim, layer_polys)
    adj_b_corners = _trim_edges(
        adj_b_corners, simple_perim, layer_polys)

    coords = np.array(simple_perim.exterior.coords)
    adj_a_path = _path_from_corners_and_coords(adj_a_corners, coords)
    adj_b_path = _path_from_corners_and_coords(adj_b_corners, coords)

    # adj_a and adj_b are the candidates for pia / wm
    pia_side, wm_side = _identify_pia_and_wm_sides(
        [adj_a_path, adj_b_path], layer_polys)

    return simple_perim, pia_side, wm_side


def _initial_vertical_side_guess(touches, tortuosities):
    max_touches = np.max(touches)
    max_mask = np.array(touches) >= max_touches
    if max_mask.sum() == 1:
        return np.flatnonzero(max_mask)[0]
    else:
        straightest_ind = np.argmin(np.array(tortuosities)[max_mask])
        return np.arange(len(touches))[max_mask][straightest_ind]


def argmin_no_duplicates(distances):
    """
    Uses Gale/Shapely stable marriage approach to find argmin of each 
    column such that no duplicates are returned. I.e. each column is a 
    groom and each row a bride, find the pairing that maximizes
    compatibility.

    Args:
        distances (2d np.array): distance of perimeter coords to the
        edge coordinates

    Returns:
        np array: row indices for each column that returns the argmin
        without duplicates
    """
    
    n = distances.shape[1]
    unmarried_grooms = list(range(n))
    engagements = {}

    while unmarried_grooms:
        groom = unmarried_grooms.pop(0)
        bride_preferences = np.argsort(distances[:, groom])
        for bride in bride_preferences:
            if bride not in engagements:
                engagements[bride] = groom
                break
            else:
                brides_current_groom = engagements[bride]
                if distances[bride, groom] < distances[bride, brides_current_groom]:
                    unmarried_grooms.append(brides_current_groom)
                    engagements[bride] = groom
                    break
    engagements = {v:k for k,v in engagements.items()}
    
    return np.array([engagements[i] for i in range(n)])


def _find_initial_corners(perimeter, boundaries):
    # Get rough estimate for ends of the top & bottom layers
    layer_order = [
        "Isocortex layer 1",
        "Isocortex layer 2/3",
        "Isocortex layer 4",
        "Isocortex layer 5",
        "Isocortex layer 6a",
        "Isocortex layer 6b",
    ]

    for l in layer_order:
        if l in boundaries:
            top_coords = boundaries[l]
            break
    for l in layer_order[::-1]:
        if l in boundaries:
            bottom_coords = boundaries[l]
            break

    top_dists = distance.squareform(distance.pdist(top_coords))
    bottom_dists = distance.squareform(distance.pdist(bottom_coords))
    top_max_inds = np.unravel_index(np.argmax(top_dists), top_dists.shape)
    bottom_max_inds = np.unravel_index(np.argmax(bottom_dists), bottom_dists.shape)

    edge_coords = np.vstack([
        top_coords[top_max_inds, :],
        bottom_coords[bottom_max_inds, :],
    ])
    perim_coords = np.array(perimeter.exterior.coords)

    dist_to_edges = distance.cdist(perim_coords[:-1, :], edge_coords)
    init_guess = np.argmin(dist_to_edges, axis=0)
    if len(np.unique(init_guess)) < len(init_guess):
        init_guess = argmin_no_duplicates(dist_to_edges)
    return init_guess


def _split_poly_into_sides(poly, corners):
    coords = np.array(poly.exterior.coords)
    corners_list = list(np.sort(corners))
    side_list = []
    corner_sets = []
    for ind_start, ind_end in zip(corners_list, corners_list[1:] + [corners_list[0]]):
        corner_sets.append((ind_start, ind_end))
        path = _path_from_corners_and_coords((ind_start, ind_end), coords)
        side_list.append(path)
    return side_list, corner_sets


def _count_layer_touches(sides, layers, tolerance=1):
    dist_list = []
    count_list = []
    for i, s in enumerate(sides):
        dist_list.append({})
        for k, l in layers.items():
            dist_list[i][k] = l.distance(s)
        count_list.append(np.sum([d < tolerance for d in dist_list[i].values()]))

    return count_list


def _tortuosity_of_sides(sides):
    tort = []
    for s in sides:
        path_length = s.length
        side_coords = np.array(s.coords)
        euc_length = distance.euclidean(side_coords[0, :], side_coords[-1, :])
        tort.append(path_length / euc_length)
    return tort


def _evaluate_moving_vertex(init_shared_vertex, init_corners, fixed_side_vertex, fixed_tb_vertex,
    step, n_vert, adj_side_ind, adj_tb_ind, shrinking_end, init_touch_diff, init_len,
    perimeter, layer_polys):

    moving_vertex = init_shared_vertex
    best_vertex = init_shared_vertex
    best_touch_diff = init_touch_diff
    best_len = init_len

    while True:
        moving_vertex += step
        moving_vertex = moving_vertex % n_vert
        if moving_vertex == shrinking_end:
            # Reached the end
            break

        curr_corners = init_corners.copy()
        curr_corners[curr_corners == init_shared_vertex] = moving_vertex
        curr_sides, curr_corner_sets = _split_poly_into_sides(perimeter, curr_corners)
        curr_touches = _count_layer_touches(curr_sides, layer_polys)

        for i, c in enumerate(curr_corner_sets):
            if moving_vertex in c:
                if fixed_side_vertex in c:
                    curr_side_candidate = i
                elif fixed_tb_vertex in c:
                    curr_tb_candidate = i
        curr_touch_diff = curr_touches[curr_side_candidate] - curr_touches[curr_tb_candidate]
        curr_len = curr_sides[curr_side_candidate].length

        if curr_touch_diff > best_touch_diff:
            best_vertex = moving_vertex
            best_touch_diff = curr_touch_diff
            best_len = curr_len
        elif (curr_touch_diff == best_touch_diff) & (curr_len < best_len):
            best_vertex = moving_vertex
            best_touch_diff = curr_touch_diff
            best_len = curr_len
    return best_vertex, best_touch_diff, best_len


def _optimize_corner_location(init_corners, sides, corner_sets, side_candidate, topbottom_candidate,
                             perimeter, layer_polys):
    side_corners = corner_sets[side_candidate]
    topbottom_corners = corner_sets[topbottom_candidate]

    shared_vertex = list(set(side_corners).intersection(topbottom_corners))[0]

    n_vert = len(perimeter.exterior.coords) - 1
    init_touches = _count_layer_touches(sides, layer_polys)
    init_tort = _tortuosity_of_sides(sides)

    # side should have more touches than top/bottom
    # first try shrinking the side
    if shared_vertex == side_corners[0]:
        step = 1
        adj_side_ind = 0
        adj_tb_ind = 1
    else:
        step = -1
        adj_side_ind = 1
        adj_tb_ind = 0

    best_shrink_vertex, best_shrink_touch_diff, best_shrink_len = _evaluate_moving_vertex(
        shared_vertex, init_corners,
        side_corners[(adj_side_ind + 1) % 2],
        topbottom_corners[(adj_tb_ind + 1) % 2],
        step, n_vert,
        adj_side_ind, adj_tb_ind,
        side_corners[(adj_side_ind + 1) % 2],
        init_touches[side_candidate] - init_touches[topbottom_candidate],
        sides[side_candidate].length,
        perimeter, layer_polys,
    )

    # then try expanding the side
    if shared_vertex == side_corners[0]:
        step = -1
        adj_side_ind = 0
        adj_tb_ind = 1
    else:
        step = 1
        adj_side_ind = 1
        adj_tb_ind = 0

    best_expand_vertex, best_expand_touch_diff, best_expand_len = _evaluate_moving_vertex(
        shared_vertex, init_corners,
        side_corners[(adj_side_ind + 1) % 2],
        topbottom_corners[(adj_tb_ind + 1) % 2],
        step, n_vert,
        adj_side_ind, adj_tb_ind,
        topbottom_corners[(adj_tb_ind + 1) % 2],
        init_touches[side_candidate] - init_touches[topbottom_candidate],
        sides[side_candidate].length,
        perimeter, layer_polys,
    )

    if best_expand_touch_diff > best_shrink_touch_diff:
        optimized_vertex = best_expand_vertex
    elif best_expand_touch_diff < best_shrink_touch_diff:
        optimized_vertex = best_shrink_vertex
    else:
        if best_expand_len < best_shrink_len:
            optimized_vertex = best_expand_vertex
        else:
            optimized_vertex = best_shrink_vertex

    opt_corners = init_corners.copy()
    opt_corners[opt_corners == shared_vertex] = optimized_vertex
    new_side_corner_set = list(side_corners)
    new_side_corner_set[adj_side_ind] = optimized_vertex

    return opt_corners, tuple(new_side_corner_set)


def _reset_sides(perim, new_corners, orig_ind, new_set):
    new_sides, new_sets = _split_poly_into_sides(perim, new_corners)

    inds = np.arange(4)
    for i, c in enumerate(new_sets):
        if c == new_set:
            new_ind = i
            break

    shift = orig_ind - new_ind
    new_inds = np.roll(inds, shift)
    sides = [new_sides[i] for i in new_inds]
    corner_sets = [new_sets[i] for i in new_inds]
    return sides, corner_sets


def _identify_pia_and_wm_sides(pia_wm_candidates, layer_polys):
    """ Find pia and wm sides

    Parameters
    ----------
    pia_wm_candidates : list of LineStrings
        Two options for pia or wm
    layer_polys : dict
        Dictionary with layer names as keys and boundary Polygons as values

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

    # Determine top (pia) side
    for l in layer_order:
        if l in layer_polys:
            top_poly = layer_polys[l]
            break

    if top_poly.distance(pia_wm_candidates[0]) < top_poly.distance(pia_wm_candidates[1]):
        pia_side = pia_wm_candidates[0]
        wm_side = pia_wm_candidates[1]
    else:
        pia_side = pia_wm_candidates[1]
        wm_side = pia_wm_candidates[0]

    return pia_side, wm_side


def _path_from_corners_and_coords(corners, coords):
    start, end = corners
    if start < end:
        path = LineString(coords[start:end + 1, :])
    else:
        combo_coords = np.vstack([coords[start:-1], coords[:end + 1]])
        path = LineString(combo_coords)
    return path


def _expand_to_touch_additional_layer(corners, simple_perim, layer_polys, tolerance=1):
    coords = np.array(simple_perim.exterior.coords)
    n_vert = coords.shape[0] - 1

    path = _path_from_corners_and_coords(corners, coords)

    # count current touches
    touches = 0
    for l in layer_polys.values():
        if l.distance(path) < tolerance:
            touches += 1

    start, end = corners

    # try expanding from the start
    current_ind = start
    found_good_expansion = False
    while True:
        next_ind = (current_ind - 1) % n_vert
        new_path = _path_from_corners_and_coords((next_ind, end), coords)
        new_touches = 0
        for l in layer_polys.values():
            if l.distance(new_path) < tolerance:
                new_touches += 1

        if new_touches == touches + 1:
            found_good_expansion = True
            break
        elif new_touches > touches + 1:
            # went too far
            break
        elif next_ind == start:
            break

        current_ind = next_ind

    if found_good_expansion:
        new_start = next_ind
    else:
        new_start = start

    # try expanding from the end
    current_ind = end
    found_good_expansion = False
    while True:
        next_ind = (current_ind + 1) % n_vert
        new_path = _path_from_corners_and_coords((next_ind, end), coords)
        new_touches = 0
        for l in layer_polys.values():
            if l.distance(new_path) < tolerance:
                new_touches += 1

        if new_touches == touches + 1:
            found_good_expansion = True
            break
        elif new_touches > touches + 1:
            # went too far
            break
        elif next_ind == end:
            break

        current_ind = next_ind

    if found_good_expansion:
        new_end = next_ind
    else:
        new_end = end
    return new_start, new_end


def _trim_edges(corners, simple_perim, layer_polys, threshold=0.50, touch_tolerance=1):
    layer_order = [
        "Isocortex layer 1",
        "Isocortex layer 2/3",
        "Isocortex layer 4",
        "Isocortex layer 5",
        "Isocortex layer 6a",
        "Isocortex layer 6b",
    ]

    # Determine top & next to top
    found_top = False
    for l in layer_order:
        if l in layer_polys and not found_top:
            top_poly = layer_polys[l]
            found_top = True
        elif l in layer_polys:
            next_to_top_poly = layer_polys[l]
            break

    # Determine bottom & next to bottom
    found_bottom = False
    for l in layer_order[::-1]:
        if l in layer_polys and not found_bottom:
            bottom_poly = layer_polys[l]
            found_bottom = True
        elif l in layer_polys:
            next_to_bottom_poly = layer_polys[l]
            break

    coords = np.array(simple_perim.exterior.coords)
    n_vert = coords.shape[0] - 1

    start, end = corners
    path = _path_from_corners_and_coords(corners, coords)

    # are we working on a top edge or bottom edge?
    if path.distance(top_poly) < path.distance(bottom_poly):
        # Working with a top edge
        nearest_poly = top_poly
        next_nearest_poly = next_to_top_poly
    else:
        # Working with a bottom edge
        nearest_poly = bottom_poly
        next_nearest_poly = next_to_bottom_poly

    # try shrinking from start
    current_end = start
    while True:
        next_to_end = (current_end + 1) % n_vert
        if next_to_end == end:
            # Can't have single-point edge
            break

        current_end_pt = Point(coords[current_end, :])
        next_to_end_pt = Point(coords[next_to_end, :])

        # First check that we haven't gone too far past the top/bottom layer - if so, drop the point
        if current_end_pt.distance(nearest_poly) > touch_tolerance:
            current_end = next_to_end
            continue

        # Get distance between end and next point
        dist_between_end_and_next = current_end_pt.distance(next_to_end_pt)

        diff_dist = next_to_end_pt.distance(next_nearest_poly) - current_end_pt.distance(next_nearest_poly)

        if diff_dist / dist_between_end_and_next < threshold:
            break
        else:
            current_end = next_to_end

    new_start = current_end

    # try shrinking from end
    current_end = end
    while True:
        next_to_end = (current_end - 1) % n_vert
        if next_to_end == new_start:
            # Can't have single-point edge
            break

        current_end_pt = Point(coords[current_end, :])
        next_to_end_pt = Point(coords[next_to_end, :])

        # First check that we haven't gone too far past the top/bottom layer - if so, drop the point
        if current_end_pt.distance(nearest_poly) > touch_tolerance:
            current_end = next_to_end
            continue

        # Get distance between end and next point
        dist_between_end_and_next = current_end_pt.distance(next_to_end_pt)

        diff_dist = next_to_end_pt.distance(next_nearest_poly) - current_end_pt.distance(next_nearest_poly)

        if diff_dist / dist_between_end_and_next < threshold:
            break
        else:
            current_end = next_to_end

    new_end = current_end

    return  new_start, new_end


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