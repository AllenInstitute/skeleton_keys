import geopandas
import logging
import shapely
import shapely.vectorized
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString
from typing import Tuple, Optional
from neuron_morphology.layered_point_depths.__main__ import setup_interpolator
from tqdm import tqdm


def cortex_thickness_aligned_y_values_for_morph(morph, avg_layer_depths,
        depth_field):
    """Point depths for a morphology aligned to the full thickness of cortex.

    Layer-specific information is not used. Zero is pia, negative values are
    below pia.

    Parameters
    ----------
    morph : Morphology
        Morphology object
    avg_layer_depths : dict
        Dictionary of distances from top of layer to pia (in microns)
    depth_field : xarray DataArray
        Data array of the depth field

    Returns
    -------
    depths_dict : dict
        Dictionary of node ids (keys) and cortex-thickness aligned depths (values)

    """
    nodes_ids, nodes_x, nodes_y = zip(*[(n["id"], n["x"], n["y"]) for n in morph.nodes()])

    nodes_ids = np.array(nodes_ids)
    nodes_x = np.array(nodes_x)
    nodes_y = np.array(nodes_y)

    depths = cortex_thickness_aligned_y_values(nodes_x, nodes_y,
        avg_layer_depths, depth_field)

    nan_mask = np.isnan(depths)
    logging.info(f"NaN y-values: {np.sum(nan_mask)}")
    return dict(zip(nodes_ids[~nan_mask], depths[~nan_mask]))



def cortex_thickness_aligned_y_values(x, y, avg_layer_depths,
        depth_field):
    """Point depths for a x, y coordinates aligned to the full thickness of cortex.

    Layer-specific information is not used. Zero is pia, negative values are
    below pia.

    Parameters
    ----------
    x : array (n_points, )
        Array of x-coordinates
    y : array (n_points, )
        Array of y-coordinates
    avg_layer_depths : dict
        Dictionary of distances from top of layer to pia (in microns) (must
        include "wm" (white matter) as a key)
    depth_field : xarray DataArray
        Data array of the depth field

    Returns
    -------
    depths : array
        Array of cortex-thickness aligned depth coordinates
    """

    # Interpolators for navigating fields - gives nan values for points outside field
    depth_interp = setup_interpolator(
        depth_field, None, method="linear",
        bounds_error=False, fill_value=None)

    pos = np.column_stack((x, y))
    rel_depths = depth_interp(pos)

    total_thickness = avg_layer_depths['wm']
    depths = -total_thickness * (1 - rel_depths)

    return depths


def layer_aligned_y_values_for_morph(morph, avg_layer_depths, layer_list,
        depth_field, gradient_field, snapped_polys_surfs):
    """Point depths for a morphology aligned to a set of layer thicknesses.

    Parameters
    ----------
    morph : Morphology
        Morphology object
    avg_layer_depths : dict
        Dictionary of distances from top of layer to pia (in microns)
    layer_list : dict
        List of layers
    depth_field : xarray DataArray
        Data array of the depth field
    gradient_field : xarray DataArray
        Data array of the depth field
    snapped_polys_surfs : dict
        Dictionary of surface and layer drawings

    Returns
    -------
    y_values : dict
        Dictionary of node ids (keys) and layer-aligned depths (values)
    """

    nodes_ids, nodes_x, nodes_y = zip(*[(n["id"], n["x"], n["y"]) for n in morph.nodes()])

    nodes_ids = np.array(nodes_ids)
    nodes_x = np.array(nodes_x)
    nodes_y = np.array(nodes_y)

    y_values_arr = layer_aligned_y_values(nodes_x, nodes_y,
        avg_layer_depths, layer_list,
        depth_field, gradient_field, snapped_polys_surfs
    )

    y_values = {n_id: y for n_id, y in zip(nodes_ids, y_values_arr) if not np.isnan(y)}
    return y_values


def layer_aligned_y_values(x, y, avg_layer_depths, layer_list,
        depth_field, gradient_field, snapped_polys_surfs):
    """Point depths for a morphology aligned to a set of layer thicknesses.

    Parameters
    ----------
    x : array (n_points, )
        Array of x-coordinates
    y : array (n_points, )
        Array of y-coordinates
    avg_layer_depths : dict
        Dictionary of distances from top of layer to pia (in microns)
    layer_list : dict
        List of layers
    depth_field : xarray DataArray
        Data array of the depth field
    gradient_field : xarray DataArray
        Data array of the depth field
    snapped_polys_surfs : dict
        Dictionary of surface and layer drawings

    Returns
    -------
    y_values : array (n_points, )
        Dictionary of node ids (keys) and layer-aligned depths (values)
    """
    # Depths of each side of layers
    avg_side_depths = layer_side_depths(avg_layer_depths, layer_list)

    # Assign nodes to layers
    layer_containing_masks = layer_locations(
        snapped_polys_surfs["polygons"], x, y)

    # QC that most nodes are actually in layers
    running_total = 0
    for k in layer_containing_masks:
        mask = layer_containing_masks[k]
        n_in_layer = np.sum(mask)
        logging.info(f"{k}: {n_in_layer}")
        running_total += n_in_layer
    logging.info(f"{running_total} out of {len(layer_containing_masks[k])} points located in layers")

    # Interpolators for navigating fields
    depth_interp = setup_interpolator_without_nan(
        depth_field, None, method="linear",
        bounds_error=False, fill_value=None)
    dx_interp = setup_interpolator_without_nan(
        gradient_field, "dx", method="linear",
        bounds_error=False, fill_value=None)
    dy_interp = setup_interpolator_without_nan(
        gradient_field, "dy", method="linear",
        bounds_error=False, fill_value=None)

    # Find intersection with layer edges to calculate fraction within layer
    surfs_dict = {s["name"]: shapely.geometry.LineString(s["path"]) for s in snapped_polys_surfs["surfaces"]}

    all_pos = np.array([x, y]).T
    y_values = np.zeros_like(y)
    mask_inside_layers = np.zeros_like(y).astype(bool)
    for layer in layer_containing_masks:
        logging.info(f"processing points in {layer}")
        mask = layer_containing_masks[layer]
        mask_inside_layers = mask_inside_layers | mask
        if np.sum(mask) == 0:
            continue

        avg_pia_side = avg_side_depths[layer + "_pia_side"]
        avg_wm_side = avg_side_depths[layer + "_wm_side"]
        pia_surf = surfs_dict[layer + "_pia"]
        wm_surf = surfs_dict[layer + "_wm"]

        pos = all_pos[mask, :]
        layer_fracs = fractions_within_layer(pos, pia_surf, wm_surf, depth_interp, dx_interp, dy_interp)
        layer_y_values = -avg_pia_side - (avg_wm_side - avg_pia_side) * layer_fracs
        if np.any(np.isinf(layer_y_values)):
            raise ValueError(f'There are inf y values in {layer}')
        y_values[mask] = layer_y_values

    mask_outside_layers = ~mask_inside_layers
    pia_coords = list(surfs_dict["pia"].coords)
    wm_coords = list(surfs_dict["wm"].coords)
    pia = surfs_dict["pia"]
    wm = surfs_dict["wm"]

    pts_in_wm = 0
    logging.info("processing points outside layers")
    outside_values = np.zeros_like(y_values[mask_outside_layers])
    for i, (pt_x, pt_y) in enumerate(all_pos[mask_outside_layers, :]):
        # check if it's within the wm
        pia_tri = shapely.geometry.Polygon([pia_coords[0], pia_coords[-1], (pt_x, pt_y)])
        wm_tri = shapely.geometry.Polygon([wm_coords[0], wm_coords[-1], (pt_x, pt_y)])
        node_pt = shapely.geometry.Point((pt_x, pt_y))
        if pia_tri.intersection(wm_tri).area > 0 and (pia_tri.area > wm_tri.area):
            # likely below white matter border, but check if it's closer to wm than pia
            if pia.distance(node_pt) < wm.distance(node_pt):
                # not in white matter since closer to pia
                outside_values[i] = np.nan
                continue
        elif wm.convex_hull.contains(node_pt):
            # in white matter, but where triangles don't overlap
            pass
        else:
            # not in white matter
            outside_values[i] = np.nan
            continue
        pts_in_wm += 1
        outside_values[i] = -avg_layer_depths["wm"] - node_pt.distance(wm)
    y_values[mask_outside_layers] = outside_values
    logging.info(f"points in wm: {pts_in_wm}")
    unassigned = len(y) - pts_in_wm - running_total
    if unassigned > 0:
        logging.info(f"{unassigned} points not assigned new y-value")

    logging.info(f"{np.sum(np.isnan(y_values))} NaN y-values")

    return y_values


def fractions_within_layer(pos, pia_side_surf, wm_side_surf,
        depth_interp, dx_interp, dy_interp, step_size=1.0, max_iter=1000):
    depths = depth_interp(pos)

    local_pia_side_depths = step_all_nodes(
        pos, depth_interp, dx_interp, dy_interp, pia_side_surf, step_size, max_iter)
    local_pia_side_depths[local_pia_side_depths > 1] = 1

    local_wm_side_depths = step_all_nodes(
        pos, depth_interp, dx_interp, dy_interp, wm_side_surf, -step_size, max_iter)
    local_wm_side_depths[local_wm_side_depths < 0] = 0

    if np.any(np.isinf(local_pia_side_depths - depths) / (local_pia_side_depths - local_wm_side_depths)):
        raise ValueError("Infinite values detected in layer fraction calculation.")

    return (local_pia_side_depths - depths) / (local_pia_side_depths - local_wm_side_depths)


def step_all_nodes(pos, depth_interp, dx_interp, dy_interp, surf, step_size, max_iter, adaptive_scale_factor=32):
    cur_pos = pos.copy()
    orig_ind = np.arange(pos.shape[0])
    adaptive_scale = np.ones(pos.shape[0]) * adaptive_scale_factor
    depths_of_intersections = np.zeros(pos.shape[0])
    finished = np.zeros(pos.shape[0]).astype(bool)

    surf_df = geopandas.GeoDataFrame({"name": ['surf'], "geometry": [surf]})

    with tqdm(total=pos.shape[0]) as pbar:
        for _ in range(max_iter):
            dx = dx_interp(cur_pos)
            dy = dy_interp(cur_pos)
            base_step = np.vstack([dx, dy]).T
            base_step = base_step / np.linalg.norm(base_step, axis=1)[:, np.newaxis]
            step = step_size * adaptive_scale[:, np.newaxis] * base_step
            step[finished] = 0 * step[finished]
            next_pos = cur_pos + step
            ray_list = [shapely.geometry.LineString([tuple(pos[i, :]), tuple(next_pos[i, :])]) for i in range(cur_pos.shape[0]) if not finished[i]]
            ray_df = geopandas.GeoDataFrame({"ind": orig_ind[~finished], "geometry": ray_list})
            intersect_df = surf_df.sjoin(ray_df, how='inner', predicate='intersects')

            # handle matches
            n_already_finished = finished.sum()
            for ind in intersect_df['ind']:
                if adaptive_scale[ind] > 1:
                    # Reduce scale and try again from same point
                    adaptive_scale[ind] /= 2
                    next_pos[ind, :] = cur_pos[ind, :]
                    continue
                ray = ray_df.set_index("ind").at[ind, "geometry"]
                intersection = ray.intersection(surf)
                if intersection.geom_type == "MultiPoint":
                    cur_pt = shapely.geometry.Point(cur_pos[ind, :])
                    dist = np.inf
                    for test_pt in intersection.geoms:
                        test_dist = cur_pt.distance(test_pt)
                        if test_dist < dist:
                            dist = test_dist
                            closest_pt = test_pt
                    intersection_pt = list(closest_pt.coords)
                else:
                    intersection_pt = list(intersection.coords)
                depths_of_intersections[ind] = float(depth_interp(intersection_pt[0]))
                finished[ind] = True
            pbar.update(finished.sum() - n_already_finished)
            if np.all(finished):
                break
            cur_pos = next_pos
    return depths_of_intersections


def layer_side_depths(avg_layer_depths,
        layer_order=["Layer1", "Layer2/3", "Layer4", "Layer5", "Layer6a", "Layer6b"]):
    sides_dict = {}
    for i, l in enumerate(layer_order):
        label = l.replace("Layer", "")
        if i == 0:
            sides_dict[l + "_pia_side"] = 0
        else:
            sides_dict[l + "_pia_side"] = avg_layer_depths[label]

        if i == len(layer_order) - 1:
            sides_dict[l + "_wm_side"] = avg_layer_depths["wm"]
        else:
            sides_dict[l + "_wm_side"] = avg_layer_depths[layer_order[i + 1].replace("Layer", "")]
    return sides_dict


def layer_locations(polys, x, y):
    is_in_layer = {}
    for p in polys:
        poly = shapely.geometry.Polygon(p["path"])
        is_in_layer[p["name"]] = shapely.vectorized.contains(poly, x, y)

    return is_in_layer


def setup_interpolator_without_nan(field, dim, **kwargs):
    coords = (field["x"].values, field["y"].values)
    if field.dims.index("x") == 1:
        coords = coords[::-1]

    if dim is None:
        values = field.values
    else:
        values = field.loc[{"dim": dim}].values

    # get rid of nan values by extending the non-nan values to edge
    for i in range(values.shape[0]):
        nz_inds = np.flatnonzero(~np.isnan(values[i, :]))
        if len(nz_inds) > 0:
            values[i, :nz_inds[0]] = values[i, nz_inds[0]]
            values[i, nz_inds[-1] + 1:] = values[i, nz_inds[-1]]
    for j in range(values.shape[1]):
        nz_inds = np.flatnonzero(~np.isnan(values[:, j]))
        if len(nz_inds) > 0:
            values[:nz_inds[0], j] = values[nz_inds[0], j]
            values[nz_inds[-1] + 1:, j] = values[nz_inds[-1], j]

    return RegularGridInterpolator(coords, values, **kwargs)


def path_dist_from_node(
    pos: Tuple[float, float],
    depth_interp: RegularGridInterpolator,
    dx_interp: RegularGridInterpolator,
    dy_interp: RegularGridInterpolator,
    surface: LineString,
    step_size: float,
    max_iter: int,
    adaptive_scale_factor: int = 32,
) -> Optional[float]:
    cur_pos = np.array(list(pos))
    pos_list = [cur_pos]
    adaptive_scale = adaptive_scale_factor
    surf_df = geopandas.GeoDataFrame({"name": ['surf'], "geometry": [surface]})

    for _ in range(max_iter):
        dx = dx_interp(cur_pos)
        dy = dy_interp(cur_pos)
        base_step = np.squeeze([dx, dy])
        base_step = base_step / np.linalg.norm(base_step)
        step = step_size * adaptive_scale * base_step
        next_pos = cur_pos + step

        ray_list = [shapely.geometry.LineString([pos, tuple(next_pos)])]
        ray_df = geopandas.GeoDataFrame({"ind": [0], "geometry": ray_list})
        intersect_df = surf_df.sjoin(ray_df, how='inner', predicate='intersects')

        # handle matches
        for ind in intersect_df['ind']:
            if adaptive_scale > 1:
                # Reduce scale and try again from same point
                adaptive_scale /= 2
                next_pos = cur_pos
                continue

            ray = ray_df.set_index("ind").at[ind, "geometry"]
            intersection = ray.intersection(surface)
            if intersection.geom_type == "MultiPoint":
                cur_pt = shapely.geometry.Point(cur_pos[ind, :])
                dist = np.inf
                for test_pt in intersection:
                    test_dist = cur_pt.distance(test_pt)
                    if test_dist < dist:
                        dist = test_dist
                        closest_pt = test_pt
                intersection_pt = list(closest_pt.coords)
            else:
                intersection_pt = list(intersection.coords)

            pos_list.append(np.array(intersection_pt))
            return calculate_length_of_path_list(pos_list)

        pos_list.append(next_pos)
        cur_pos = next_pos

    return None


def calculate_length_of_path_list(path_list):
    pos_arr = np.vstack(path_list)

    deltas = np.diff(pos_arr, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    total_dist = dists.sum()
    return total_dist
