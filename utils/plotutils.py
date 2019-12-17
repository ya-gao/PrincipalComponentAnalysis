# -*- coding: utf-8 -*-
# @Author: jesse
# @Date:   2018-09-20 15:44:03
# @Last Modified by:   jesse
# @Last Modified time: 2018-10-05 16:27:41

import matplotlib as mpl
import progutils as pru

gwfdir = pru.get_maindir()
mpl.rcParams['font.family'] = "Times New Roman"
mpl.rcParams['figure.autolayout'] = True


# ----------------------------------------------------------------------------
# BASE PLOTS
# ----------------------------------------------------------------------------
def plot_lake(ax=None, fig=None, lake=None, subbasins=False, fill=False):
    import glob
    import numpy as np
    import matplotlib as mpl
    from matplotlib.collections import PatchCollection

    if ax is None:
        fig, ax = mpl.pyplot.subplots(figsize=(12, 6.75), dpi=160)

    # Based off specified lake, find shapefile we need
    if lake is None:
        shpdir = "{}shp/GreatLakes/".format(gwfdir)
    elif "erie" in lake.lower():
        if "west" in lake.lower():
            shpdir = "{}shp/Lake_Erie/Western/".format(gwfdir)
        elif "east" in lake.lower():
            shpdir = "{}shp/Lake_Erie/Eastern/".format(gwfdir)
        elif "cent" in lake.lower():
            shpdir = "{}shp/Lake_Erie/Central/".format(gwfdir)
        else:
            shpdir = "{}shp/Lake_Erie/".format(gwfdir)
    elif "ont" in lake.lower():
        shpdir = "{}shp/Lake_Ontario/".format(gwfdir)
    elif "hur" in lake.lower():
        shpdir = "{}shp/Lake_Huron/".format(gwfdir)
    elif "mic" in lake.lower():
        shpdir = "{}shp/Lake_Michigan/".format(gwfdir)
    elif "sup" in lake.lower():
        shpdir = "{}shp/Lake_Superior/".format(gwfdir)
    else:
        print("I'M NOT READY FOR THAT YET")

    # Find shapediles (assume there could be more than 1)
    shpfiles = glob.glob("{}*.shp".format(shpdir))

    polygons = []
    colours = []
    draworder = []

    # Loop over shapefiles
    for shpfile in shpfiles:
        if not subbasins and 'subbasin' in shpfile.lower():
            continue
        elif subbasins and 'subbasin' in shpfile.lower():
            tmppolys = unpack_shp(shpfile)
            polygons.extend(tmppolys)
            colours.extend([0.1] * len(tmppolys))
            draworder.extend([0] * len(tmppolys))
        else:
            tmppolys = unpack_shp(shpfile)
            polygons.extend(tmppolys)
            islands = [is_island(tmppolys, i) for i in tmppolys]
            colours.extend([0.1 if i else 0.35 for i in islands])
            draworder.extend([2 if i else 1 for i in islands])

    drawinds = np.argsort(draworder)
    polygons = [polygons[i] for i in drawinds]
    colours = [colours[i] for i in drawinds]

    patchcoll = PatchCollection(
        polygons,
        color='white',
        cmap=mpl.cm.ocean,
        edgecolor='k',
        lw=0.5
    )
    patchcoll.set_clim(0, 1)
    if fill:
        patchcoll.set_array(np.array(colours))
    ax.add_collection(patchcoll)

    minlon, minlat, maxlon, maxlat = poly_bbox(polygons)
    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)

    return fig, ax, polygons


def plot_stations(fig, ax, datdict, region=None, lake=None):

    fig, ax, polygons = plot_lake(ax=ax, lake=lake)

    c = getcolour(region)

    if region is not None:
        label = "{} Basin".format(region)
    else:
        label = ""

    ax.scatter(
        datdict['lon'],
        datdict['lat'],
        label=label,
        color=c,
        zorder=3
    )
    ax.set_title("In Situ Measurement Locations")
    ax.legend()


# ----------------------------------------------------------------------------
# POLYGON TOOLS
# ----------------------------------------------------------------------------
def is_island(polygons, testpoly):
    for p in polygons:
        if p == testpoly:
            continue
        path = p.get_path()

        if all(path.contains_points(testpoly.get_path().vertices)):
            return True
    return False


def unpack_shp(shpfile):
    import shapefile
    from matplotlib.patches import Polygon
    polygons = []
    shpfile = shapefile.Reader(shpfile)

    for shp in shpfile.iterShapes():
        parts = shp.parts
        points = shp.points
        if len(parts) == 1:
            polygons.append(Polygon(points, zorder=0))
        else:
            for ind, startind in enumerate(parts[:-1]):
                polygons.append(
                    Polygon(points[startind:parts[ind + 1]], zorder=0)
                )

    return polygons


def poly_itersect(poly1, poly2, entire=False):
    if entire:
        return is_island([poly1], poly2)
    path = poly1.get_path()
    points = poly2.get_path().vertices
    return any(path.contains_points(points))


def poly_bbox(polygons):
    if not isinstance(polygons, list):
        polygons = list(polygons)
    allminlon = []
    allminlat = []
    allmaxlon = []
    allmaxlat = []

    for p in polygons:
        vertices = p.get_path().vertices
        tmplon = [i[0] for i in vertices]
        tmplat = [i[1] for i in vertices]
        allminlon.append(min(tmplon))
        allminlat.append(min(tmplat))
        allmaxlon.append(max(tmplon))
        allmaxlat.append(max(tmplat))
    return min(allminlon), min(allminlat), max(allmaxlon), max(allmaxlat)


def crop_overpoly(xx, yy, zz, polygons):
    import numpy as np

    tmpzz = zz.ravel()
    islands = [is_island(polygons, p) for p in polygons]

    for ind, point in enumerate(zip(xx.ravel(), yy.ravel())):
        for p, island in zip(polygons, islands):
            if island and p.get_path().contains_point(point):
                tmpzz[ind] = np.nan
            elif not island and not p.get_path().contains_point(point):
                tmpzz[ind] = np.nan
    return tmpzz.reshape(zz.shape)


def grab_borderpoints(points, polygons):
    import numpy as np
    keepers = []

    islands = [is_island(polygons, p) for p in polygons]

    lons, lats = zip(*points)
    for ind, p in enumerate(polygons):
        if islands[ind]:
            continue
        for point in p.get_path().vertices:
            dists = haversine_arr(lons, lats, *point)
            if np.sum(dists <= 3) == 0:
                continue
            else:
                keepind = np.argmin(dists)
                keepers.append(keepind)
    return keepers


# ----------------------------------------------------------------------------
# PLOT FORMATTING MADE EASY
# ----------------------------------------------------------------------------
def easy_axformat(
    fig,
    ax,
    title="",
    xlabel="",
    ylabel="",
    xlim=None,
    ylim=None,
    titsize=14,
    lblsize=12,
    ticksize=12,
    leg=False,
    leg_fs=12,
    grid=False,
    gridwhich='major',
    xtime=None
):
    mpl_years = mpl.dates.YearLocator()   # every year
    mpl_months = mpl.dates.MonthLocator()  # every month

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(b=grid, which=gridwhich)
    if leg:
        ax.legend(fontsize=leg_fs)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if xtime is not None:
        ax.xaxis.set_major_locator(
            mpl_years if "y" in xtime.lower() else mpl_months
        )
        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%Y' if "y" in xtime.lower() else '%m')
        )

    # Fontsizing
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label, "axes"]:
        if item == ax.title:
            item.set_fontsize(titsize)
        elif item == ax.xaxis.label or item == ax.yaxis.label:
            item.set_fontsize(lblsize)
        else:
            ax.tick_params(axis='both', labelsize=ticksize)


def getcolour(region):
    if region == "Western":
        c = 'green'
    elif region == "Central":
        c = 'red'
    elif region == "Eastern":
        c = 'blue'
    elif region == 'St. Clair':
        c = 'yellow'
    else:
        c = 'k'
    return c


def get_plotparams(dkey):
    if dkey == 'chla':
        minval = 0
        maxval = 25
        title = r'Chlorophyll-A ($mg/m^3$)'
    elif dkey == 'ox':
        minval = 4
        maxval = 16
        title = r'Dissolved Oxygen ($mg/L$)'
    elif dkey == 'phos':
        minval = 0.
        maxval = .08
        title = r'Phosphorus ($mg/L$)'
    elif dkey == 'chli':
        minval = 10
        maxval = 20
        title = r'Chloride ($mg cL/L$)'
    return minval, maxval, title


# ----------------------------------------------------------------------------
# DISTANCES AND POINT CHECKING
# ----------------------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    from math import radians, sin, cos, asin, sqrt
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def haversine_arr(lons, lats, chklon, chklat):
    from numpy import radians, sin, cos, arcsin, sqrt
    lons = radians(lons)
    chklon = radians(chklon)
    lats = radians(lats)
    chklat = radians(chklat)
    dlon = chklon - lons
    dlat = chklat - lats

    a = sin(dlat / 2)**2 + cos(lats) * cos(chklat) * sin(dlon / 2)**2
    c = 2 * arcsin(sqrt(a))
    km = 6371 * c
    return km


def closest_point(chkpoints, point, thres=None):
    import numpy as np

    lons, lats = zip(*chkpoints)
    lons = np.array(lons)
    lats = np.array(lats)

    dgrid = haversine_arr(lons, lats, *point)

    if thres is None:
        return np.argmin(dgrid)
    else:
        return np.where(dgrid <= thres)[0]


# ----------------------------------------------------------------------------
# PLOTTING TOOLS
# ----------------------------------------------------------------------------
def unpack_xy(datdict, ykey='chla'):
    import numpy as np
    import matplotlib as mpl

    xpoints = np.array(
        [mpl.dates.date2num(i) for i in datdict['date']]
    )
    ypoints = np.array(datdict[ykey])
    return xpoints, ypoints


def prep_interp(xpoints, ypoints, kind='quadratic'):
    from scipy.interpolate import interp1d
    import numpy as np

    f = interp1d(xpoints, ypoints, kind=kind)
    newx = np.arange(min(xpoints), max(xpoints), 1)
    newy = f(newx)

    return newx, newy


def smooth(y, window, poly):
    from scipy.signal import savgol_filter

    return savgol_filter(y, window, poly)


def SOM(
    points,
    values,
    radius,
    rate,
    iterations,
    minlon,
    minlat,
    maxlon,
    maxlat,
    binning=0.02,
    polygons=None,
    bar=True
):
    import itertools
    import numpy as np

    # Initialise grid:
    xx, yy = np.meshgrid(
        np.arange(minlon, maxlon + binning, binning),
        np.arange(minlat, maxlat + binning, binning)
    )
    grid = zip(xx.ravel(), yy.ravel())
    gridlen = len(grid)
    inlen = len(values)
    startvals = np.random.uniform(low=0., high=max(values), size=(gridlen,))
    if polygons is not None:
        startvals = crop_overpoly(xx, yy, startvals, polygons)

    pointcyc = itertools.cycle(points)
    valcyc = itertools.cycle(values)
    touched = np.full(startvals.shape, 0)
    for it in xrange(iterations):

        if bar and it == 0:
            pru.progress(it, iterations)

        dist = radius * np.exp(-float(it)**2 / (1e7 * iterations))
        dist = dist if dist >= 4.6 else 4.6
        adjust = rate * np.exp(-float(it) / (iterations))

        # First find which data point we're comparing to
        ind = np.random.randint(inlen)

        # Now unpack data
        chkpoint = next(pointcyc)
        refval = next(valcyc)

        if np.isnan(refval):
            continue

        # generate all the distances from reference point
        closest_inds = closest_point(grid, chkpoint, thres=dist)
        touched[closest_inds] += 1
        if closest_inds.size == 0:
            continue
        closest_ind = closest_inds[0]
        if startvals[closest_ind] > refval:
            startvals[closest_ind] -= adjust * refval
        elif startvals[closest_ind] < refval:
            startvals[closest_ind] += adjust * refval

        neigh_inds = closest_inds[1:]
        neigh_lons, neight_lats = zip(*np.array(grid)[neigh_inds])
        dists = haversine_arr(neigh_lons, neight_lats, *chkpoint)

        for ind, tmpdist in zip(neigh_inds, dists):
            if np.isnan(startvals[ind]):
                continue
            tmpval = startvals[ind]
            if tmpval > refval:
                startvals[ind] -= adjust * (
                    1 - tmpdist / float(dist)
                ) * refval
            elif tmpval < refval:
                startvals[ind] += adjust * (
                    1 - tmpdist / float(dist)
                ) * refval

        if bar:
            pru.progress(it + 1, iterations)

    startvals[touched <= 0.15 * iterations] = np.nan
    startvals = startvals.reshape(xx.shape)

    return xx, yy, startvals
