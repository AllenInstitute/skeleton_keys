import shapely
import numpy as np
from functools import partial
from neuron_morphology.snap_polygons.geometries import (
    Geometries, make_scale, clear_overlaps, closest_from_stack,
    get_snapped_polys, find_vertical_surfaces, select_largest_subpolygon
)
from neuron_morphology.transforms.geometry import get_vertices_from_two_lines


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
    bounds = shapely.geometry.polygon.Polygon(pia_wm_vertices)

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
