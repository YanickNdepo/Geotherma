import re
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from IPython.display import HTML, display
from utils.config import DEFAULT_TILES
import folium as flm
from folium import plugins


def jupyter_cell_bkgd(color):
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}" style="display:none">'.format(script)))


def dataframe_viewer(df, rows=10, cols=12, step_r=1, step_c=1, un_val=None, view=True):
    # display dataframes with  a widget

    if un_val is None:
        print(f'Rows : {df.shape[0]}, columns : {df.shape[1]}')
    else:
        if isinstance(un_val, str):
            un_val = [un_val]

        for c in un_val:
            if c in df.columns:
                len(set(df[c]))
        print(f"Rows : {df.shape[0]}, columns : {df.shape[1]}, "
              f"Unique values on cols: {dict({c: len(set(df[c])) if c in df.columns else 'NA' for c in un_val})}")

    if view:
        @interact(last_row=IntSlider(min=min(rows, df.shape[0]), max=df.shape[0],
                                     step=step_r, description='rows', readout=False,
                                     disabled=False, continuous_update=True,
                                     orientation='horizontal'),
                  last_column=IntSlider(min=min(cols, df.shape[1]),
                                        max=df.shape[1], step=step_c,
                                        description='columns', readout=False,
                                        disabled=False, continuous_update=True,
                                        orientation='horizontal')
                  )
        def _freeze_header(last_row, last_column):
            display(df.iloc[max(0, last_row - rows):last_row,
                    max(0, last_column - cols):last_column])


def dict_viewer(dictionary):
    """
    Jupyter Notebook magic repr function for dictionaries.
    """
    rows = ''
    s = '<td>{v}</td>'
    ss = '<td>{vk}</td><td>{vv}</td>'
    for k, v in dictionary.items():
        if isinstance(v, dict):
            for vk, vv in v.items():
                cels = ss.format(vk=vk, vv=vv)
                rows += '<tr><td><strong>{k}</strong></td>{cels}</tr>'.format(k=k, cels=cels)
        else:
            cels = s.format(v=v)
            rows += '<tr><td><strong>{k}</strong></td>{cels}</tr>'.format(k=k, cels=cels)

    html = '<table>{}</table>'.format(rows)
    return display(HTML(html))


def geodf_map(geodf, plot=True, inter_plot=False, _return=False, **kwargs):
    """
        create a map and compute gobal limits from one or many geodataframes

        Parameters
        -----------
        geodf (list) : a (or a list of) Geopandas.GeoDataframe objects
        _return (bool) : if True, return values of global bounds of the area and alongsides X, Y distances

        kwargs:
            expand (tuple) : expand global limits by values specified (x, y); reduced if values are negative
            shift (tuple) : shift global limits by values specified (x, y)
            colors (list): list of colors names for each geodataframe to plot
            figsize (tuple)

        returns
        --------
        bounds (list) : global area limits like [(xmin, xmax), (ymin, ymax)]
        dist_side_xy (tuple) : distance alongsides X and Y
    """

    expand = kwargs.pop('expand', None)
    shift = kwargs.pop('shift', None)
    colors = kwargs.pop('colors', None)
    opac = kwargs.pop('alpha', None)
    figsize = kwargs.pop('figsize', None)
    lname = kwargs.pop('layers_name', [f'shapefile_{i}' for i in range(len(geodf))])

    if not isinstance(geodf, list):
        geodf = [geodf]

    xmin, xmax, ymin, ymax = [], [], [], []
    for gdf in geodf:
        xmin.append(min(gdf.bounds.minx))
        xmax.append(max(gdf.bounds.maxx))
        ymin.append(min(gdf.bounds.miny))
        ymax.append(max(gdf.bounds.maxy))

    # another way : merge all gdf and use .total_bounds()
    bounds = [(min(xmin), max(xmax)), (min(ymin), max(ymax))]

    if isinstance(expand, tuple):
        bounds[0] = (min(bounds[0]) - expand[0], max(bounds[0]) + expand[0])
        bounds[1] = (min(bounds[1]) - expand[1], max(bounds[1]) + expand[1])

    if isinstance(shift, tuple):
        bounds[0] = tuple(np.array(bounds[0]) - shift[0])
        bounds[1] = tuple(np.array(bounds[1]) - shift[1])

    limits = [(x, y) for x in bounds[0] for y in bounds[1]]
    x = [x[0] for x in limits]
    y = [x[1] for x in limits]
    xlim = [min(bounds[0]), min(bounds[0]), min(bounds[0]), max(bounds[0]), max(bounds[0]), min(bounds[0])]
    ylim = [min(bounds[1]), max(bounds[1]), max(bounds[1]), max(bounds[1]), min(bounds[1]), min(bounds[1])]

    if plot :
        if not inter_plot:
            plt.figure(figsize=figsize)
            ax = plt.subplot(111)
            if opac is None:
                opac = [.5 for i in range(len(colors))]
            elif isinstance(opac, float) or isinstance(opac, int):
                opac = [opac for i in range(len(colors))]

            for n, gdf in enumerate(geodf): gdf.plot(ax=ax, color=colors[n], alpha=opac[n])
            plt.plot(xlim, ylim, color='k', linestyle='dotted')
            #plt.scatter(x, y, color='red', marker='x')
            plt.show()
        else:
            _map = None
            for n, gdf in enumerate(geodf):
                if _map is None:
                    _map = gdf.explore(color=colors[n], name=lname[n])
                else:
                    gdf.explore(m=_map, color=colors[n], name=lname[n])

            flm.TileLayer('openstreetmap', control=True).add_to(_map)  # use folium to add alternative tiles
            flm.LayerControl().add_to(_map)  # use folium to add layer control

    dist_side_xy = (max(bounds[0]) - min(bounds[0]), max(bounds[1]) - min(bounds[1]))
    if inter_plot and _return:
        return _map, bounds, dist_side_xy
    elif _return:
        return bounds, dist_side_xy
    elif inter_plot:
        return _map


def geodf_readf(files, coords_cols=['X', 'Y'], epsg=None, to_epsg=None, overwrite=False, sep=','):
    """
    create a geodataframe from a list of files and transform coordinates system (if 'to_epsg' is set)

    Parameters
    -------------
    files: list
        list of filename (.gpkg, .shp, .json, .csv)

    coords_cols : list
        columns name for coordinates

    epsg: int
        actual data Coordinates EPSG number

    to_epsg : int
        coordinates EPSG number to convert into


    Returns
    ---------
    geodataframe object

    """

    gdf_list = []
    for filename in files:
        if filename is None:
            filename = str(input("File name and extension (.shp, .json, .gpkg, .csv) ? : "))

        if re.compile(r".+\.json|.+\.gpkg|.+\.shp").match(filename):
            with open(filename, 'r'):
                gdf = gpd.read_file(filename)

        if re.compile(r".+\.csv").match(filename):
            if epsg is None:
                epsg = input("data EPSG (a number) ? : ")

            with open(filename, 'r'):
                df = pd.read_csv(filename, header=0, sep=sep)

                if 'geometry' in df.columns:
                    df['geometry'] = df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=str(f"EPSG:{epsg}"))

                elif (coords_cols[0] and coords_cols[1]) in [x.lower() for x in df.columns]:
                    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[coords_cols[0]], df[coords_cols[1]],
                                                                           crs=str(f"EPSG: {epsg}")))
                else:
                    print("Error, the input dataframe does not have the correct coordinates fields !!")

        # EPSG conversion

        if to_epsg is not None and overwrite:
            gdf.to_crs(epsg=to_epsg, inplace=True)

            while True:  # use infinte loop to force get in loop
                resp = str(input("Overwrite old coordinates fields ? (y/n) : ")).strip().lower()
                if resp == 'y':
                    gdf = gdf.drop(['Longitude', 'Latitude', 'X', 'Y'], axis=1, errors='ignore')
                    gdf.insert(0, 'X', [row._geometry.x for idx, row in gdf.iterrows()])
                    gdf.insert(1, 'Y', [row._geometry.y for idx, row in gdf.iterrows()])
                    break
                elif resp == 'n':
                    break
                print(f'{resp} is invalid, please try again...')

        elif to_epsg is not None and overwrite is False:
            gdf.to_crs(epsg=to_epsg, inplace=True)
            gdf = gdf.drop(['Longitude', 'Latitude', 'X', 'Y'], axis=1, errors='ignore')
            gdf.insert(0, 'X', [row._geometry.x for idx, row in gdf.iterrows()])
            gdf.insert(1, 'Y', [row._geometry.y for idx, row in gdf.iterrows()])

        gdf_list.append(gdf)
    return gdf_list


def geodf_export(gdf, epsg, save_name=None):
    """
    Save data location in a geodataframe into Geopackage / GeoJson / csv file

    Parameters
    -----------
    gdf: geopandas.GeoDataframe object
        a dataframe from which we build the geodataframe

    epsg: int
        Coordinates EPSG number to be saved

    save_name: str
        file's name and extension format (.gpkg, .json, .csv)
    """

    if save_name is None:
        save_name = str(input("File name and extension (.json, .gpkg, .csv) ? : "))

    gdf.to_crs(epsg=str(epsg), inplace=True)
    ext = save_name[save_name.rfind('.') + 1:]
    if ext == 'json':
        gdf.to_file(f'{save_name}', driver="GeoJSON")
        print(f'{save_name}' + " has been saved !")
    elif ext == 'gpkg':
        gdf.to_file(f'{save_name}', driver="GPKG", layer="Boreholes")
        print(f'{save_name}' + " has been saved !")
    elif ext == 'csv':
        if 'X' in gdf.columns:
            gdf = gdf.drop(['X'], axis=1)
            gdf.insert(0, 'X', gdf._geometry.x)
        else:
            gdf.insert(0, 'X', gdf._geometry.x)

        if 'Y' in gdf.columns:
            gdf = gdf.drop(['Y'], axis=1)
            gdf.insert(1, 'Y', gdf._geometry.y)
        else:
            gdf.insert(1, 'Y', gdf._geometry.y)

        gdf.to_csv(f'{save_name}', index_label="Id", index=False, sep=',')
        print(f'{save_name}' + " has been saved !")
    else:
        print(f'file\'s name extension not given or incorrect, please choose (.json, .gpkg, .csv)')


def gdf_map(gdf, id_col='ID', tiles=None, epsg=31370, save_as=None, radius=0.5, opacity=1, zoom_start=15, max_zoom=25,
             control_scale=True, marker_color='red'):
    """2D Plot of all boreholes in the project

    parameters
    -------------
    tile : List of dicts containing tiles properties (name, attributes, url)
    epsg : int
        Value of Coordinates Reference System (CRS)
    save_as : str
         filename (and dir) to save html version of the map (e.g: 'mydir/mymap.html')

    """

    if not 'geometry' in gdf.columns:
        geom = gpd.points_from_xy(gdf.X, gdf.Y, crs=f"EPSG:{epsg}")
        gdf.geometry = geom
        gdf.drop(columns=['X', 'Y'], inplace=True)

    # Change CRS EPSG 31370 (Lambert 72) into EPSG 4326 (WGS 84)
    if epsg != 4326:
        gdf = gdf.to_crs(epsg=4326)
    center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]

    # Use a satellite map
    if tiles is None:
        tiles = DEFAULT_TILES

    global_map = flm.Map(location=center, tiles='OpenStreetMap', zoom_start=zoom_start,
                     max_zoom=max_zoom, control_scale=control_scale)

    ch1 = flm.FeatureGroup(name='Boreholes')

    for idx, row in gdf.iterrows():
        flm.CircleMarker([row.geometry.y, row.geometry.x], popup=row[id_col],
                        radius=radius, color=marker_color, fill_color=marker_color,
                        opacity=opacity).add_to(ch1)
        # flm.map.Marker([row.geometry.y, row.geometry.x], popup=row.Name).add_to(ch1)

    mini_map = plugins.MiniMap(toggle_display=True, zoom_level_offset=-6)

    # adding features to the base_map
    for tile in tiles:
        flm.TileLayer(name=tile['name'], tiles=tile['url'], attr=tile['attributes'],
                 max_zoom=max_zoom, control=True).add_to(global_map)

    ch1.add_to(global_map)
    flm.LayerControl().add_to(global_map)
    global_map.add_child(mini_map)

    # save in a file
    if save_as is not None:
        global_map.save(save_as)  # ('tmp_files/BH_location.html')

    return global_map