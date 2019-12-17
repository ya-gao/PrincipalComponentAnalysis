import utils.progutils as pru
import matplotlib as mpl

gwfdir = pru.get_maindir()
mpl.rcParams['font.family'] = "Times New Roman"
mpl.rcParams['figure.autolayout'] = True


# ============================================================================
# LAKE SHAPE CLASS
# ============================================================================

class lake:
    '''
    Lake Object.
    Reads in a shapefile to polygons and organizes them.
    Contains useful functions for working with data over lakes
    '''

    def __init__(self, lake=None, subbasin=False, bboxpad=0):
        shpfiles = get_lakefiles(lake=lake)

        polygons = []
        subbasin_polys = []
        for shpfile in shpfiles:
            if 'subbasin' in shpfile.lower() and not subbasin:
                continue
            elif 'subbasin' in shpfile.lower() and subbasin:
                subbasin_polys.extend(unpack_shp(shpfile))
                continue
            polygons.extend(unpack_shp(shpfile))

        isl_bool = [is_island(p, polygons) for p in polygons]

        self.basin = subbasin_polys
        self.islands = [p for isl, p in zip(isl_bool, polygons) if isl]
        self.body = [p for isl, p in zip(isl_bool, polygons) if not isl]
        self.polygons = polygons
        self.isl_bool = isl_bool

        self.bbox = poly_bbox(
            self.body if not self.basin else self.basin,
            pad=bboxpad
        )

        # Generate paths now to save runtime later.
        # A little memory heavy but possibly worth it
        self.bpath = [i.get_path() for i in self.body]
        self.ipath = [i.get_path() for i in self.islands]
        self.polycover = []

    def plot(self, fig=None, ax=None, fill=False):
        '''
        Plot the polygons onto an axis. Assumes cyl eq area projection
        '''
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
            give = 2
        elif fig is None and ax is not None:
            give = 0
        else:
            give = 0

        patchcoll = []
        colors = []

        # First add basins
        patchcoll.extend(self.basin)
        colors.extend([0.1] * len(self.basin))

        # Next add lake body
        patchcoll.extend(self.body)
        colors.extend([0.35] * len(self.body))

        # Lastly Islands:
        patchcoll.extend(self.islands)
        colors.extend([0.1] * len(self.islands))

        patchcoll = mpl.collections.PatchCollection(
            patchcoll,
            color='white',
            cmap=mpl.cm.ocean,
            edgecolor='k',
            lw=0.5
        )
        patchcoll.set_clim(0, 1)

        if fill:
            patchcoll.set_array(np.array(colors))

        ax.add_collection(patchcoll)
        minlon, minlat, maxlon, maxlat = self.bbox
        ax.set_xlim(minlon, maxlon)
        ax.set_ylim(minlat, maxlat)

        if give == 0:
            return
        elif give == 2:
            return fig, ax
        else:
            raise RuntimeError

    def contains(self, points, borders=True, bthres=4.6):
        '''
        Check if point of points lie within the lake but not over island
        '''
        import numpy as np
        if not isinstance(points, list):
            points = list(points)

        out_bool = np.full((len(points),), False)
        if borders:
            border_bool = self.bordering(points, thres=bthres)

        # First eliminate points not in body:
        for b in self.bpath:
            inbody = b.contains_points(points)
            out_bool[np.where(inbody)[0]] = True
        for i in self.ipath:
            inisland = np.array(i.contains_points(points))
            out_bool[np.where(inisland)[0]] = False

        if borders:
            out_bool = (out_bool | border_bool)

        return list(out_bool)

    def crop(self, zz, xx=None, yy=None, borders=True, bthres=4.6):
        '''
        Given grid and data, it crops out data not in lake
        '''
        import numpy as np

        zzshape = zz.shape
        x = np.linspace(self.bbox[0], self.bbox[2], zzshape[1], endpoint=True)
        y = np.linspace(self.bbox[1], self.bbox[3], zzshape[0], endpoint=True)
        if xx is None and yy is None:
            xx, yy = np.meshgrid(x, y)
        elif xx is not None and yy is not None:
            pass
        else:
            raise RuntimeError
        points = zip(xx.ravel(), yy.ravel())
        mask = np.array(self.contains(points, borders=borders, bthres=bthres))
        mask = mask.reshape(zz.shape)
        zz[~mask] = np.nan

        return zz

    def bordering(self, points, thres=3):
        '''
        Given list of points, it finds which are within a certain
        distance from lake shore
        '''
        import numpy as np

        out_bool = np.full((len(points),), False)

        for b in self.bpath:
            for v in b.vertices:
                keepers = closest_point(points, v, thres=thres)
                out_bool[keepers] = True

        return list(out_bool)

    def cover(self):
        '''
        WIP:
        Meant to build a cover for areas around the lake,
        so if they are present they are not visible.
        Purely for aesthetic and not efficient
        '''
        import numpy as np
        minlon, minlat, maxlon, maxlat = self.bbox

        for b in self.bpath:
            vertices = b.vertices
            minind = np.where(vertices[:, 0] == minlon)[0]
            maxind = np.where(vertices[:, 0] == maxlon)[0]
            print(vertices[minind])
            print(minlon)
            print(vertices[maxind])
            print(maxlon)

    def rough_crop(self, zz, xx=None, yy=None):
        import numpy as np

        zzshape = zz.shape
        x = np.linspace(self.bbox[0], self.bbox[2], zzshape[1], endpoint=True)
        y = np.linspace(self.bbox[1], self.bbox[3], zzshape[0], endpoint=True)
        if xx is None and yy is None:
            xx, yy = np.meshgrid(x, y)
        elif xx is not None and yy is not None:
            pass
        else:
            raise RuntimeError
        points = zip(xx.ravel(), yy.ravel())

        out_bool = np.full((len(points),), False)
        for b in self.bpath:
            keepers = b.contains_points(points)
            out_bool[keepers] = True

        zz[~out_bool.reshape(zzshape)] = np.nan

        return zz


# ============================================================================
# SHAPE FILE AND POLYGON FUNCTIONS
# ============================================================================
def get_lakefiles(lake=None):
    '''
    Gets lakefiles needed based off specified lake key
    '''
    import glob
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

    return shpfiles


def unpack_shp(shpfile):
    '''
    Read in a shapefile and unpack it into Polygons
    '''
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


def is_island(testpoly, polygons):
    '''
    given list of polygons, and a test polygon,
    it determines whether or not polygon is island to list of polygons
    '''
    for p in polygons:
        if p == testpoly:
            continue
        path = p.get_path()

        if all(path.contains_points(testpoly.get_path().vertices)):
            return True
    return False


def poly_bbox(polygons, pad=0):
    '''
    Determines bounding box for the polygon
    (All the minimums and maximums)
    '''
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
    return min(allminlon) - pad, min(allminlat) - pad, max(allmaxlon) + pad, max(allmaxlat) + pad


# ============================================================================
# DISTANCE AND GEOMETRY
# ============================================================================
def haversine_arr(lons, lats, chklon, chklat):
    '''
    given list of latitudes and longitudes
    as well as a test coordinate, it determines the approximate
    distance between those two coordinate assuming the earth
    is a sphere
    '''
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
    '''
    Determines the closest point (possibly within some threshold)
    to another point on the sphere
    '''
    import numpy as np

    lons, lats = zip(*chkpoints)
    lons = np.array(lons)
    lats = np.array(lats)

    dgrid = haversine_arr(lons, lats, *point)

    if thres is None:
        return np.argmin(dgrid)
    else:
        return np.where(dgrid <= thres)[0]


# ============================================================================
# PLOT FORMATTING
# ============================================================================
def easy_axformat(
    fig,
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    xlim=None,
    ylim=None,
    titsize=24,
    lblsize=20,
    ticksize=20,
    leg=False,
    leg_fs=14,
    grid=False,
    gridwhich='major',
    xtime=None
):
    import matplotlib as mpl
    mpl_years = mpl.dates.YearLocator()   # every year
    mpl_months = mpl.dates.MonthLocator()  # every month

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_axisbelow(True)
    ax.grid(b=grid, which=gridwhich)

    if leg:
        ax.legend(fontsize=leg_fs)
    if not leg:
        tmpleg = ax.legend([])
        tmpleg.remove()

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
        if 'y' in xtime.lower():
            ax.tick_params(axis='x', rotation=45)

    # Fontsizing
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label, "axes"]:
        if item == ax.title:
            item.set_fontsize(titsize)
        elif item == ax.xaxis.label or item == ax.yaxis.label:
            item.set_fontsize(lblsize)
        else:
            ax.tick_params(axis='both', labelsize=ticksize)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)


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
        c = 'blue'
    return c


def get_units(dkey):
    if dkey == 'chla':
        return 'mg/m3'
    elif dkey in ['lwst', 'airtemp']:
        return 'Degrees Celsius'
    elif dkey == 'lat':
        return 'Degrees North'
    elif dkey == 'lon':
        return 'Degrees East'
    elif dkey in ['precip', 'snow']:
        return 'mm'
    elif dkey == 'wind':
        return 'm/s'
    elif dkey == 'cloud':
        return '0-1'
    elif dkey == 'inflow':
        return 'm3/s'
    elif dkey == 'water_lev':
        return 'm'
    elif dkey in ['sus_sed', 'phos']:
        return 'mg/s'
    elif dkey == 'ice':
        return '%'
    else:
        return None


def get_name(dkey):
    if dkey == 'chla':
        return 'Chlorophyll-a'
    elif dkey == 'lwst':
        return "LWST"
    elif dkey == 'airtemp':
        return 'Air Temperature'
    elif dkey == 'lat':
        return 'Latitude'
    elif dkey == 'lon':
        return 'Longitude'
    elif dkey == 'precip':
        return 'Precipitation'
    elif dkey == 'snow':
        return 'Snow'
    elif dkey == 'wind':
        return 'Wind'
    elif dkey == 'cloud':
        return 'Cloud Cover'
    elif dkey == 'inflow':
        return 'Inflow'
    elif dkey == 'water_lev':
        return 'Water Level'
    elif dkey == 'sus_sed':
        return 'Suspended Sediment'
    elif dkey == 'phos':
        return 'Phosphorus'
    elif dkey == 'ice':
        return 'Ice'
    elif dkey == 'PH':
        return 'pH'
    else:
        return None


def get_plotparams(dkey):
    if dkey == 'chla':
        minval = 0
        maxval = 30
        title = r'Chlorophyll-A ($mg/m^3$)'
    elif dkey == 'oxy':
        minval = 4
        maxval = 16
        title = r'Dissolved Oxygen ($mg/L$)'
    elif dkey == 'phos':
        minval = 0.
        maxval = .08
        title = r'Phosphorus ($mg/L$)'
    elif dkey == 'clf':
        minval = 10
        maxval = 30
        title = r'Chloride ($mg/L$)'
    else:
        return [None] * 3
    return minval, maxval, title


# ============================================================================
# MAPPING DATA
# ============================================================================
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
    lake=None,
    bar=True
):
    import itertools
    import numpy as np

    # Initialise grid:
    xx, yy = np.meshgrid(
        np.arange(minlon, maxlon + binning, binning),
        np.arange(minlat, maxlat + binning, binning)
    )
    grid = list(zip(xx.ravel(), yy.ravel()))
    inlen = len(values)
    startvals = np.random.uniform(low=0., high=max(values), size=xx.shape)
    if lake is not None:
        startvals = lake.crop(startvals, xx=xx, yy=yy, bthres=10)
    startvals = startvals.ravel()
    pointcyc = itertools.cycle(points)
    valcyc = itertools.cycle(values)
    touched = np.full(startvals.shape, 0)
    for it in range(iterations):

        if bar and it == 0:
            pru.progress(it, iterations)

        dist = radius * np.exp(-1. / iterations * np.floor((it + 1.) / inlen))
        dist = dist if dist >= 4.6 else 4.6
        adjust = rate * np.exp(-1. / iterations * np.floor((it + 1.) / inlen))

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

    touchthres = round(0.15 * iterations)
    touchthres = touchthres if touchthres <= 150 else 150
    startvals[touched <= touchthres] = np.nan
    startvals = startvals.reshape(xx.shape)

    return xx, yy, startvals


def year_fraction(date):
    import datetime as dt
    start = dt.datetime(date.year, 1, 1).toordinal()
    year_length = dt.datetime(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


# ============================================================================
# QUICK PLOTS
# ============================================================================
def quick_timeseries(
    srcdata,
    dkey,
    fig=None,
    ax=None,
    sep=True,
    region=None,
    xtime='y',
    ylabel=None,
    err=False,
    mk=False
):
    import utils.dictutils as du
    import matplotlib.pyplot as plt
    import matplotlib.dates as mpld
    import numpy as np

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)

    regions = ["Western", "Central", "Eastern"]

    if sep:
        for tmpdict, region in zip(du.separate_regions(srcdata), regions):

            c = getcolour(region)
            tmpdict = du.flatten_dict(tmpdict)
            xpoints = mpld.date2num(tmpdict['date'])
            trendx = np.array([year_fraction(i) for i in tmpdict['date']])

            ypoints = np.array(tmpdict[dkey])
            keepers = np.isfinite(ypoints)

            xpoints = xpoints[keepers]
            trendx = trendx[keepers]
            ypoints = ypoints[keepers]

            if err:
                yerr = (1 - np.array(tmpdict['chla_cov'])[keepers]) * ypoints
                ax.errorbar(
                    xpoints,
                    ypoints,
                    yerr,
                    color=c,
                    capsize=2,
                    alpha=0.5,
                    ls=''
                )

            trendy, m, b, p = trend_analysis(
                trendx,
                ypoints,
                yerr=None if not err else yerr,
                mk=mk
            )

            label = "{} Basin (slope: {:.2g}, p: {:.2f})".format(
                region, m, p
            )

            minval, maxval, ylabel = get_plotparams(dkey)

            if any([i is None for i in [minval, maxval, ylabel]]):
                minval = np.nanmin(ypoints)
                maxval = np.nanmax(ypoints)
                ylabel = None

            ax.plot(xpoints, ypoints, 'o--', color=c, alpha=0.5)
            ax.plot(xpoints, trendy, color=c, lw=2, label=label)

    elif not sep:
        c = getcolour(region)

        tmpdict = du.flatten_dict(srcdata)
        xpoints = mpld.date2num(tmpdict['date'])
        trendx = np.array([year_fraction(i) for i in tmpdict['date']])

        ypoints = np.array(tmpdict[dkey])
        keepers = np.isfinite(ypoints)

        xpoints = xpoints[keepers]
        trendx = trendx[keepers]
        ypoints = ypoints[keepers]

        if err:
            yerr = (1 - np.array(tmpdict['chla_cov'])[keepers]) * ypoints
            ax.errorbar(
                xpoints, ypoints, yerr, color=c, capsize=2, alpha=0.5, ls=''
            )

        trendy, m, b, p = trend_analysis(
            trendx,
            ypoints,
            yerr=None if not err else yerr,
            mk=mk
        )

        if region is None:
            label = "slope: {:.2g}, p: {:.2f}".format(m, p)
        else:
            label = "{} Basin (slope: {:.2g}, p: {:.2f})".format(
                region, m, p
            )
        minval, maxval, ylabel = get_plotparams(dkey)

        if any([i is None for i in [minval, maxval, ylabel]]):
            minval = np.nanmin(ypoints)
            maxval = np.nanmax(ypoints)
            ylabel = ""

        ax.plot(xpoints, ypoints, 'o--', color=c, alpha=0.5)
        ax.plot(xpoints, trendy, color=c, lw=2, label=label)

    easy_axformat(
        fig,
        ax,
        leg=True,
        xtime=xtime,
        grid=True,
        ylabel="{} ({})".format(
            dkey, srcdata['units'][0]
        ) if ylabel is None else ylabel
    )

    return fig, ax


def linear(p, x):
    m, b = p
    return m * x + b


def trend_analysis(trendx, trendy, yerr=None, mk=False):
    import numpy as np
    import utils.mk_test as mk
    from scipy.stats import linregress

    trendx = np.array(trendx)
    trendy = np.array(trendy)

    trend, h, p, z = mk.mk_test(trendy)

    estm, estb = linregress(trendx, trendy)[:2]

    if h and estm > 0 and trend == 'decreasing':
        raise RuntimeError
    elif h and estm < 0 and trend == 'increasing':
        raise RuntimeError

    fit_y = estm * trendx + estb

    return list(fit_y), estm, estb, p
