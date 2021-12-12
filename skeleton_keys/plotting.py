import seaborn as sns
import numpy as np


def plot_cortical_cell(ax, sk, ld=None, title=None):
    """plot a cortical neuron according to ivscc style plot

    Args:
        ax (matplotlib.axes]): axes in which to put plot
        sk (allensdk.core.swc.Morphology): skeleton file read with allensdk
          assumes the skeleton has negative y values which get more
          negative as the cell gets lower in cortex, and has
          compartment labels with soma as 2
        ld (dict): dictionary of layer depths, where values are positive
                   indicating how below pia (assumed to be at zero)
                   layers are, note will be inverted for plotting purposes
        title ([type]): what title to give the plot
    """
    MORPH_COLORS = {3: "firebrick", 4: "salmon", 2: "steelblue"}
    for compartment, color in MORPH_COLORS.items():
        lines_x = []
        lines_y = []
        guess = None
        for c in sk.compartment_list_by_type(compartment):
            if c["parent"] == -1:
                continue
            p = sk.compartment_index[c["parent"]]
            lines_x += [p["x"], c["x"], None]
            lines_y += [p["y"], c["y"], None]
        ax.plot(lines_x, lines_y, c=color, linewidth=0.5)
        ax.set_aspect("equal")
    #     plt.gca().invert_yaxis()

    if ld is not None:
        depths = [k for k in ld.values()]
        depths += [0.0]
        depths = np.array(depths) * -1

        ax.hlines(depths, xmin=-300, xmax=300, linestyles="dashed", color="gray")
        ax.set_xlim(-300, 300)
    sns.despine(left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_layer_polygon(ax, surfaces_and_paths):
    pia_surface = surfaces_and_paths["pia_path"]
    wm_surface = surfaces_and_paths["wm_path"]
    soma_drawing = surfaces_and_paths["soma_path"]
    layer_polygons = surfaces_and_paths["layer_polygons"]

    path = np.array(pia_surface["path"]) * pia_surface["resolution"]
    ax.plot(path[:, 0], path[:, 1])
    path = np.array(wm_surface["path"]) * pia_surface["resolution"]
    ax.plot(path[:, 0], path[:, 1])
    path = np.array(soma_drawing["path"]) * pia_surface["resolution"]
    ax.plot(path[:, 0], path[:, 1])
    for poly in layer_polygons:
        path = np.array(poly["path"]) * pia_surface["resolution"]
        ax.plot(path[:, 0], path[:, 1])
