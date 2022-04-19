import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import flopy as fp
import warnings
import os
from shapely.geometry import Point, Polygon, LineString
from flopy.utils.gridgen import Gridgen


def features_intersect(model, data=None, layer=None, local=False, intersect=True, surf_intersect=True, gridgen=None, verbose=False):
    """
        Intersects a feature ('point', 'line', 'polygon') with a model grid or returns all model grid
        cells nodes coordinates

        Parameters
        ------------
        model : Modflow.model object
            a gwf or gwt model
        data : list
            list of (shp or gpkg files, GeoDataFrame or shapely.geometry objects)
        layer : int or list
            force feature intersection with specific layer(s)
        local : bool
            if True, use cells local coordinates
        intersect : bool
            if True, intersect model grid with features list in 'data'; if False, simply returns all model grid nodes coordinates
        surf_intersect: bool
            if True, polygon's surface is considered for the intersection
        gridgen: flopy.utils.gridgen.Gridgen object

        returns
        --------
        node_coords (list): list of intersected cells nodes coordinates like [node, (l,r,c), (x,y,z)]
    """

    mgrid = model.modelgrid
    intv = np.unique(mgrid.top_botm)[::-1]
    if data is None:
        print("No data, only returns model cells coordinates")
        intersect = False

    def _cells_coordinates(mgrid, local):
        xyz = mgrid.xyzcellcenters
        cells_coords = [[mgrid.get_node(mgrid.intersect(x, y, z, local=local))[0],
                         mgrid.intersect(x, y, z, local=local), (x, y, z)] for z in np.unique(xyz[2])
                        for x in np.unique(xyz[0]) for y in np.unique(xyz[1])]
        return cells_coords

    def _polygon_process(ftr, gridgen, layer, verbose):
        isect_nodes = []  # intersected nodes number
        isect_lay = []
        points = [i for i in ftr.exterior.coords]
        maxdepth = np.nanmin([p[2] for p in points])  # max depth reached by one of the feature elements

        # find layer based on max depth of the feature points
        if layer is not None:
            layer = list(layer) if not isinstance(layer, (int, float)) else [layer]
        else:
            layer = range(gridgen.get_nlay())
        """else:
            intv = np.unique(gridgen.modelgrid.top_botm)[::-1]
            for i in range(len(intv) - 1):
                a, b = intv[i], intv[i + 1]
                if min(a, b) <= maxdepth < max(a, b):
                    lk = i
                    break"""
        for lk in layer:
            try:
                nodes = gridgen.intersect(features=[ftr], featuretype=ftr.type.lower(), layer=lk)
            except Exception:
                print(f'intersection not possible for model layer {lk}')
            print(nodes[0])
            for node in nodes:
                isect_nodes.append(node[0])
            if nodes and lk not in isect_lay: isect_lay.append(lk)

        if verbose: print(f'Total of {len(isect_nodes)} cells intersected in following Layers : {isect_lay}')
        return isect_nodes

    def _process(feature, nodes_list, gridgen, surf_intersect, layer, verbose):
        gdg_msg = "A gridgen is needed to intersect surfaces. If None, set polygon_intersect='border'"
        coords = []
        if isinstance(feature, Polygon):
            if not surf_intersect:
                coords = feature.exterior.coords[:]
            else:
                assert isinstance(gridgen, Gridgen), gdg_msg
                nodes_list.extend(_polygon_process(feature, gridgen, layer, verbose))
        else:
            coords = feature.coords[:]

        for p in coords:
            if len(p) == 3:
                x, y, z = p
            elif len(p) == 2:
                x, y = p
                z = 0
            else:
                # case with M values
                pass

            if layer is not None:
                for i in range(len(intv) - 1):
                    if i == layer:
                        z = np.mean([intv[i], intv[i + 1]])
                        break

            lrc = [mgrid.intersect(x, y, z, local=local)]
            nodes_list.append(mgrid.get_node(lrc)[0])
        return nodes_list

    def _intersect(mgrid, data, layer, local, surf_intersect, verbose):
        d_msg = 'Only shp or gpkg files, GeoDataFrame or Point, LineString, Polygon objects are allowed'

        gdf_list, plp_list, gdf_nodes = [], [], []
        assert data is not None, 'Data not given'
        if not isinstance(data, list):
            data = [data]

        for dat in data:
            assert isinstance(dat, (str, Point, LineString, Polygon, gpd.GeoDataFrame)), d_msg
            if isinstance(dat, str):
                assert dat.split('.')[1] in ['gpkg', 'shp'], "File formats allowed are: 'gpkg', 'shp'"
                if os.path.isfile(dat):
                    gdf_list.append(gpd.read_file(dat))
            elif isinstance(dat, gpd.GeoDataFrame):
                gdf_list.append(dat)
            elif isinstance(dat, (Point, LineString, Polygon)):
                plp_list.append(dat)

        for d in gdf_list:
            for i in d.index:
                ftr = d.geometry[i]
                gdf_nodes = _process(ftr, gdf_nodes, gridgen, surf_intersect, layer, verbose)

        for ftr in plp_list:
            gdf_nodes = _process(ftr, gdf_nodes, gridgen, surf_intersect, layer, verbose)

        gdf_nodes = list(set(gdf_nodes))
        nodes_coords = [c for c in _cells_coordinates(mgrid, local) if c[0] in gdf_nodes]

        return nodes_coords

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if intersect:
            nodes_coords = _intersect(mgrid, data, layer, local, surf_intersect, verbose)
        else:
            nodes_coords = _cells_coordinates(mgrid, local)
        w = list(filter(lambda i: issubclass(i.category, UserWarning), w))
        if len(w):
            print('\033[48;5;225m', w[0].message)  # show warning once

    return nodes_coords


def model_map(model, plot_array=None, cnt_array=None, layer_line=None, plot_type='map', annotations=None, show_grid=True,
              show_vectors=False, shapefiles=None, **kwargs):
    """
        Plot one or many MODFLOW model's 2D maps at once

        Parameters
        -----------
        layer_line (list) : list of the layers or lines (cross-sections) to show
        cnt_array (numpy.array) : contour array to draw contours
        plot_array (numpy.array) : another data array different from data (residual, ...)
        plot_type (str) : 'map', 'row' or 'column' (for cross_section)
        shapefiles (list) : List of dict of shapefiles (and theirs properties) to add to the figure.
            e.g : shp = {'shp':--, 'ax'=ax, 'kw'={}}
        annotations (dict) : Dict containing annotations properties (nodes coordinates, format, color, marker, ...).
            - nodes coordinates must be like layer-row-column and XYZ coordinates --> e.g: [[node, (l,r,c), (x,y,z)], ...]
            - shifting of the annotations must be a tuple -->  (x,y)
            e.g : annotations = {'coords':[[10, (0,1,1), (2500,2500)], ...], 'shift': (0,0), 'text': [8, 'r'], 'marker': ['.', 'b', 5]}
    """

    font_dict = kwargs.pop('fontdict', {'fontsize': 15, 'fontweight': 'normal', 'color': 'red'})
    figsize = kwargs.pop('figsize', (20, 5))
    cnt_label = kwargs.pop('cnt_label', 10)
    cnt_fmt = kwargs.pop('cnt_fmt', "%2.5f")
    cnt_col = kwargs.pop('cnt_color', 'black')
    cb_fmt = kwargs.pop('cb_fmt', None)
    cb_shrink = kwargs.pop('cb_shrink', .5)
    alph = kwargs.pop('alpha', 1)
    cmap = kwargs.pop('cmap', None)
    sub_fig_col = kwargs.pop('fig_col', 3)
    sb_kw = kwargs.pop('subplot_kw', {'aspect': 'equal'})
    hspace = kwargs.pop('hspace', None)
    wspace = kwargs.pop('wspace', None)
    const_lyt = kwargs.pop('constrained_layout', True)
    gc = kwargs.pop('grid_color', 'k')
    glw = kwargs.pop('grid_lw', .3)
    layer_cdt = kwargs.pop('consider_layer', True)
    model_pkg = kwargs.pop('gwf_pkg', 'CHD') # e.g: 'CHD', 'WEL', 'GHB', ...

    if hspace is not None or wspace is not None: const_lyt = False
    if cnt_array is None:
        cnt_array = plot_array
    if cnt_array is not None:
        n_cnt = kwargs.pop('n_cnt', 10)
        _min, _max = np.nanmin(cnt_array), np.nanmax(cnt_array)
        _step = (_max - _min)/n_cnt
        cnt_itv = kwargs.pop('cnt_intervals', np.arange(_min, _max, _step))

    mg = model.modelgrid
    Nlay, Nrow, Ncol = mg.shape

    tb = np.unique(mg.top_botm)
    if tb[0] < 0: tb = tb[::-1]  # list inversion
    top_botm = [f'(top: {tb[n]} ; botm: {tb[n + 1]})' for n in range(len(tb) - 1)]

    cross_step = kwargs.pop('cross_step', round((Nrow-1)/4))
    layer_list = []
    if layer_line is None:
        if plot_type.lower() == 'map':
            layer_list = list(range(Nlay))
        elif plot_type.lower() in ['row', 'column']:
            print(f"Show only {plot_type} cross-sections stepped by {cross_step} !")
            layer_list = list(range(0, Nrow, cross_step))
    else:
        if isinstance(layer_line, (int, float)):
            layer_list = [int(layer_line)]
        elif hasattr(layer_line, '__iter__'):
            layer_list = layer_line
        else:
            raise (TypeError("'Layer_line' parameter must be an int or a list of int"))

    if not layer_cdt:  # output only one plot if not considering layers or rows
        layer_list = [0]
        sub_fig_col = 1

    Nplot = len(layer_list)  # Number of plots
    if Nplot <= sub_fig_col:
        sub_fig_row = 1
        sub_fig_col = Nplot
    else:
        add_row = 1 if (Nplot % sub_fig_col) != 0 and (Nplot % sub_fig_col) < sub_fig_col else 0
        sub_fig_row = int(Nplot / sub_fig_col) + add_row

    fs = figsize
    if plot_type.lower() == 'map':
        fig, axes = plt.subplots(sub_fig_row, sub_fig_col, figsize=(fs[0], fs[1] * sub_fig_row), constrained_layout=const_lyt, subplot_kw=sb_kw)
    elif plot_type.lower() in ['row', 'column']:
        if sub_fig_col > 2:  # For cross-sections visibility, figure column is reduced to 2 !
            sub_fig_col = 2
            add_row = 1 if (Nplot % sub_fig_col) != 0 and (Nplot % sub_fig_col) < sub_fig_col else 0
            sub_fig_row = int(Nplot / sub_fig_col) + add_row
        fig, axes = plt.subplots(sub_fig_row, sub_fig_col, figsize=(fs[0], fs[1]), constrained_layout=const_lyt)
    else:
        raise(ValueError("plot_type must be 'map', 'row', 'column'"))

    k = 0
    fg_r = 0
    for i in layer_list:
        for fg_c in range(0, sub_fig_col):
            if k >= len(layer_list): break
            lk = layer_list[k]
            if not isinstance(axes, np.ndarray):
                ax = axes
            else:
                if sub_fig_col == 1 or len(axes.shape) < 2:
                    ax = axes[k]
                else:
                    ax = axes[fg_r, fg_c]

            if plot_type.lower() == 'map':
                if lk >= Nlay: raise(ValueError("Layer's number out of range"))
                ax.set_title(f"Layer {lk} {top_botm[lk]}", fontdict=font_dict)
                modelmap = fp.plot.PlotMapView(model=model, ax=ax, layer=lk)
            elif plot_type.lower() in ['row', 'column']:
                ax.set_title(f"{plot_type.capitalize()} {lk}", fontdict=font_dict)
                modelmap = fp.plot.PlotCrossSection(model=model, ax=ax, line={plot_type: lk})
                lk = list(range(0, plot_array.shape[0]))

            if plot_array is not None:
                vmin = kwargs.pop('vmin', np.nanmin(plot_array))
                vmax = kwargs.pop('vmax', np.nanmax(plot_array))
                if layer_cdt:
                    par = plot_array[lk]
                    cnt_ar = cnt_array[lk]
                else:
                    par = plot_array
                    cnt_ar = cnt_array
                pa = modelmap.plot_array(par, vmin=vmin, vmax=vmax, alpha=alph, cmap=cmap)
                try:  # catch this : AttributeError: 'NoneType' object has no attribute 'parent'
                    modelmap.plot_bc(model_pkg)
                except(AttributeError):
                    break
                plt.colorbar(pa, shrink=cb_shrink, ax=ax, format=cb_fmt)
            if cnt_array is not None:
                contours = modelmap.contour_array(a=cnt_ar, levels=cnt_itv, colors=cnt_col)
                ax.clabel(contours, fmt=cnt_fmt, fontsize=cnt_label)
            if show_grid:
                modelmap.plot_grid(linewidth=glw, color=gc)
            if annotations is not None:
                lay = annotations['layer'] if 'layer' in annotations.keys() else None
                node_coords = annotations['coords'] if 'coords' in annotations.keys() else None
                sha = annotations['shift'] if 'shift' in annotations.keys() else (-np.unique(mg.delc)[0]/6, np.unique(mg.delr)[0]/6)
                txt = annotations['text_fmt'] if 'text_fmt' in annotations.keys() else None
                mk = annotations['marker'] if 'marker' in annotations.keys() else [['.', 'b', 15]]
                an_typ = annotations['annot_type'] if 'annot_type' in annotations.keys() else 'node'  # ['node', 'rc']

                if node_coords is not None:
                    if not isinstance(mk[0], list):
                        mk = [mk]
                    if not isinstance(node_coords[0][0], list):
                        node_coords = [node_coords]
                    assert len(node_coords) <= len(mk), "Marker's list must be same length or higher than coords list"
                    if lay is None:
                        lay = list(set([n[1][0] for j in range(len(node_coords)) for n in node_coords[j]]))
                    elif not hasattr(lay, '__iter__'):
                        lay = [lay]

                    for j in range(len(node_coords)):
                        for n in node_coords[j]:
                            if n[1][0] in lay and n[1][0] == lk:
                                ax.scatter(n[2][0], n[2][1], marker=mk[j][0], c=mk[j][1], s=mk[j][2])
                                if txt is not None:
                                    if an_typ == 'rc':
                                        txt_val = n[1][1:]
                                    elif an_typ == 'node':
                                        txt_val = mg.get_node(n[1])[0]
                                    ax.annotate(txt_val, (n[2][0] + sha[0], n[2][1] + sha[1]), size=txt[0], color=txt[1])
            k += 1
        fg_r += 1
        if not const_lyt: fig.subplots_adjust(hspace=hspace, wspace=wspace)  # it not seems to work !!

        # ---- add shapefiles to map (to be improved or written elsewhere) -----

        if shapefiles is not None and isinstance(shapefiles, list):
            for shp in shapefiles:
                assert isinstance(shp, dict)
                _kw = {'radius': 500.0, 'cmap': 'Dark2', 'edgecolor': 'scaled', 'facecolor': 'scaled','a': None,
                       'masked_values': None, 'idx': None}

                fp.plot.plot_shapefile(shp=shp['shp'], ax=shp.pop('ax', ax), **shp.pop('kw', _kw))
