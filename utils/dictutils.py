import utils.progutils as pru

gwfdir = pru.get_maindir()


# ----------------------------------------------------------------------------
# DATASET AND DATA OBJECT CLASS
# ----------------------------------------------------------------------------
class datset(dict):
    '''
    A Class to generalize input data to scripts
    '''

    def __init__(self, filetype, region=None):
        '''
        __init__ with a specific filetype so source of data can be traced.
        OPTIONAL: Include a region if doing an Erie subbasin
        '''
        self.filetype = filetype
        self.region = None

    def __eq__(self, other):
        '''
        For comparisons between other datasets
        '''
        if self.dkeys() != other.dkeys():
            return False

        for key in self.dkeys():
            if not all(self[key] == other[key]):
                return False
        return True

    def dkeys(self):
        '''
        Returns what data is present in the dataset
        Sorted in alphabetical order
        '''
        return sorted(self.keys())

    def remove(self, years=None, months=None, depths=None):
        '''
        Remove specified years then months.
        If either is none, nothing will be removed
        '''
        import numpy as np

        # Iterate and remove years if years inputted
        if years is not None:
            # Ensure input is list for iterative capability
            years = list(years) if not isinstance(years, list) else years
            for dkey in self.dkeys():
                keepers = np.array(
                    [i.year not in years for i in self[dkey]['date']]
                )
                tmpdf = self[dkey][keepers]
                tmpdf.units = self[dkey].units
                self[dkey] = tmpdf
        # Iterate over months in months inputted
        if months is not None:
            # Ensure input is list for iterative capability
            months = list(months) if not isinstance(months, list) else months
            for dkey in self.dkeys():
                keepers = np.array(
                    [i.month not in months for i in self[dkey]['date']]
                )
                tmpdf = self[dkey][keepers]
                tmpdf.units = self[dkey].units
                self[dkey] = tmpdf
        # Iterate over Depths and remove those not wanted
        if depths is not None:
            depths = list(depths) if not isinstance(depths, list) else depths
            for dkey in self.dkeys():
                keepers = np.full(len(self[dkey]['depth']), False)
                for depth in depths:
                    keepers = keepers | (
                        np.abs(self[dkey]['depth'] - depth) <= 5
                    )
                tmpdf = self[dkey][~keepers]
                tmpdf.units = self[dkey].units
                self[dkey] = tmpdf

    def copy(self):
        '''
        Returns an exact duplicate of the dataset so it can be manipulated
        without altering the larger dataset
        '''
        import copy
        tmp = copy.deepcopy(self)
        for key in self.dkeys():
            tmp[key].units = self[key].units
        return tmp

    def crop(self, years=None, months=None, depths=None):
        '''
        Specify what to keep, the rest will be cut away.
        Depths should be entered in multiples of 10
        '''

        # Iterate and remove years if years inputted
        if years is not None:
            tmpyears = list(range(1900, 2100))
            # Ensure input is list for iterative capability
            years = [years] if not isinstance(years, list) else years
            for year in years:
                tmpyears.pop(tmpyears.index(year))
        else:
            tmpyears = None

        if months is not None:
            tmpmonths = list(range(1, 13))
            # Ensure input is list for iterative capability
            months = [months] if not isinstance(months, list) else months
            for month in months:
                tmpmonths.pop(tmpmonths.index(month))
        else:
            tmpmonths = None

        if depths is not None:
            tmpdepths = list(range(0, 1010, 10))
            depths = [depths] if not isinstance(depths, list) else depths
            for depth in depths:
                tmpdepths.pop(tmpdepths.index(depth))
        else:
            tmpdepths = None

        self.remove(years=tmpyears, months=tmpmonths, depths=tmpdepths)

        return

    def append(self, filename, filetype, newdat=None):
        '''
        Insert a new dataset into the current one
        '''
        import numpy as np
        import utils.betaplotutils as pu

        # Generate new dataset
        if newdat is None:
            newdat = beta_parsefile(filename, filetype)

        # Iterate over new available data and add it to current
        for dkey in newdat.dkeys():
            if dkey not in self.dkeys():
                continue
            newdat[dkey]['source'] = np.full(
                len(newdat[dkey]['value']), newdat.filetype
            )
            self[dkey]['source'] = np.full(
                len(self[dkey]['value']), self.filetype
            )
            self[dkey] = self[dkey].append(newdat[dkey], sort=True)
            self[dkey].sort_values('date', inplace=True)
            self[dkey] = self[dkey].reset_index(drop=True)
            self[dkey].units = newdat[dkey].units

            tmpdf = self[dkey]
            cols = list(tmpdf)
            cols.append(cols.pop(cols.index('source')))
            self[dkey] = tmpdf.reindex(columns=cols)
            self[dkey].units = pu.get_units(dkey)

    def average(
        self,
        timeres="Y",
        keep=['loc', 'source', 'depth'],
        maxval=False
    ):
        '''
        WORK IN PROGRESS:
        Average into varying time groups.
        Flatten by year, yearmonth, or month
        '''
        import numpy as np
        import datetime as dt
        import utils.betaplotutils as pu

        # Loop over all internal variables
        for ind, dkey in enumerate(self.dkeys()):
            tmpdf = self[dkey]

            # Change all 'dates' to just the year
            if timeres.lower() == "y":
                sortgroup = np.array([i.year for i in tmpdf['date']])
                undostr = "%Y"
            elif timeres.lower() == "ym":
                sortgroup = np.array(
                    ["{:%Y%m}".format(i) for i in tmpdf['date']]
                )
                undostr = "%Y%m"
            elif timeres.lower() == "m":
                sortgroup = np.array([i.month for i in tmpdf['date']])
                undostr = "%m"
            elif timeres.lower() == "ymd":
                sortgroup = np.array([
                    "{:%Y%m%d}".format(i) for i in tmpdf['date']
                ])
                undostr = "%Y%m%d"
            else:
                raise RuntimeError("Invalid resolution for Averaging")

            # Group and Average
            tmpdf['date'] = sortgroup

            sortcols = ['date']

            if 'lat' in list(tmpdf) and 'loc' in keep:
                sortcols += ['lat', 'lon']
            elif 'lat' not in list(tmpdf):
                pass
            else:
                tmpdf.drop(['lat', 'lon'], axis=1, inplace=True)

            if 'source' in list(tmpdf) and 'source' in keep:
                sortcols += ['source']
            elif 'source' not in list(tmpdf):
                pass
            else:
                tmpdf.drop(['source'], axis=1, inplace=True)

            if 'depth' in list(tmpdf) and 'depth' in keep:
                sortcols += ['depth']
            elif 'depth' not in list(tmpdf):
                pass
            else:
                tmpdf.drop(['depth'], axis=1, inplace=True)
            if not maxval:
                tmpdf = tmpdf.groupby(
                    sortcols, as_index=False, sort=True
                ).mean()
            else:
                tmpdf = tmpdf.groupby(
                    sortcols, as_index=False, sort=True
                ).max()

                if tmpdf.size == 0:
                    continue

            tmpdf['date'] = np.array(
                [dt.datetime.strptime(str(i), undostr) for i in tmpdf['date']]
            )
            # Averaging causes loss of units, this just brings them back
            tmpdf.units = pu.get_units(dkey)

            # Replace original Dataset
            self[dkey] = tmpdf

    def timeseries(
        self,
        dkey,
        xtime='y',
        coverr=False,
        trend=True,
        fig=None,
        ax=None,
        c=None,
        depth=None,
        output=None
    ):
        '''
        For performing a quick timeseries of specified data onto an axis
        If axis is not given, one will be generated.
        Returns figure and axis object even when input
        '''
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import utils.betaplotutils as pu

        # Make Sure Axis Exists
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)

        # Initialize xpoints and ypoints removing non-finite values
        ypoints = self[dkey]['value'][:]
        if len(ypoints) <= 3:
            print("RuntimeWarning: Not enough datapoints, returning.")
            return fig, ax
        keepers = np.isfinite(ypoints)

        # If depth filtering is on, also do that
        if depth is not None and 'depth' in list(self[dkey]):
            keepers = keepers & (np.abs(self[dkey]['depth'] - depth) <= 5)

        xpoints = mpl.dates.date2num(self[dkey]['date'][:])[keepers]
        ypoints = ypoints[keepers]
        if coverr and 'cov' in list(self[dkey]):
            yerr = (1 - self[dkey]['cov'][keepers]) * ypoints

        # Put some values into raw year values so a yearly slope is obtained
        yearfracs = np.array(
            [pu.year_fraction(i) for i in self[dkey]['date'][keepers]]
        )

        # Set colour based off Region, if possible
        if self.region is not None and c is None:
            c = pu.getcolour(self.region)
        elif c is not None:
            pass
        else:
            c = 'k'

        # Determine y axis bounds and label
        minval, maxval, ylabel = pu.get_plotparams(dkey)
        if minval is None or maxval is None or ylabel is None:
            minval = min(ypoints)
            maxval = max(ypoints)
            if hasattr(self[dkey], 'units'):
                ylabel = "{} ({})".format(
                    pu.get_name(dkey),
                    self[dkey].units
                )
            else:
                ylabel = "{}".format(pu.get_name(dkey))

        # Plot Basic Timeseries
        if 'source' in list(self[dkey]):
            eckeepers = self[dkey]['source'] == 'eccc'
            epakeepers = self[dkey]['source'] == 'epa'

            if np.sum(eckeepers) != 0:
                ecval = self[dkey]['value'][:][eckeepers]
                ecdate = mpl.dates.date2num(self[dkey]['date'][:])[eckeepers]
                ax.plot(
                    ecdate,
                    ecval,
                    marker='s',
                    ms=4,
                    zorder=5,
                    alpha=0.75,
                    ls=None,
                    color=c,
                    lw=0,
                    label=""
                )
            if np.sum(epakeepers) != 0:
                epaval = self[dkey]['value'][:][epakeepers]
                epadate = mpl.dates.date2num(self[dkey]['date'][:])[epakeepers]
                ax.plot(
                    epadate,
                    epaval,
                    marker='^',
                    ms=4,
                    zorder=5,
                    alpha=0.75,
                    ls=None,
                    color=c,
                    lw=0,
                    label=""
                )
            if ax.get_legend() is None:
                ax.plot(
                    [],
                    [],
                    color='k',
                    marker='^',
                    label="EPA",
                    ls=None,
                    lw=0
                )
                ax.plot(
                    [],
                    [],
                    color='k',
                    marker='s',
                    label="ECCC",
                    ls=None,
                    lw=0
                )

        else:
            if coverr:
                ax.errorbar(
                    xpoints,
                    ypoints,
                    fmt='o',
                    yerr=yerr,
                    color=c,
                    alpha=0.5 if trend else 1,
                    label=""
                )
            else:
                ax.plot(
                    xpoints,
                    ypoints,
                    'o',
                    color=c,
                    alpha=0.5 if trend else 1,
                    label=""
                )

        # Make trendx empty so it doesn;t appear on plot
        trendx = yearfracs
        trendy = ypoints
        if trend:
            fit_y, m, b, p = pu.trend_analysis(trendx, trendy)
            label = "slope: {:.2g}, p: {:.2f}".format(
                m, p
            ) if (
                self.region is None
            ) else "{} Basin (slope: {:.2g}, p: {:.2f})".format(
                self.region, m, p
            )
        else:
            label = ""

        # Plot trendline.
        # If trend is false, this occurs just to generate label
        ax.plot(
            xpoints if trend else [],
            fit_y if trend else [],
            color=c,
            label=label,
            lw=1.4
        )

        # Format axes
        minx, maxx = ax.get_xlim()
        if minx > min(xpoints) - 180:
            minx = min(xpoints) - 180
        if maxx < max(xpoints) + 180:
            maxx = max(xpoints) + 180
        xlim = (minx, maxx)
        pu.easy_axformat(
            fig,
            ax,
            xlim=xlim,
            leg=trend,
            xtime=xtime,
            ylabel=ylabel,
            grid=True
        )

        if output is not None:
            import pandas as pd
            import os

            if os.path.exists(output):
                outdict = pd.read_csv(output).to_dict(orient='list')

            else:
                outdict = {
                    'Region': [],
                    'Depth': [],
                    'N': [],
                    'm': [],
                    'p': []
                }
            if self.region is not None:
                outdict['Region'].append(self.region)
            else:
                outdict['Region'].append('Not Specified')

            outdict['Depth'].append(depth)
            outdict['N'].append(len(ypoints))
            outdict['m'].append(round(m, 2))
            outdict['p'].append(round(p, 2))

            outstr = pd.DataFrame(
                data=outdict
            ).reindex(
                ['Region', 'Depth', 'N', 'm', 'p'], axis=1
            ).to_csv(
                index=False
            )

            with open(output, 'w+') as f:
                f.write(outstr)

        return fig, ax

    def anomaly(
        self,
        dkey,
        c=None,
        yerr=False,
        trend=False,
        fig=None,
        ax=None,
        offset=0,
        bwidth=70,
        diff=True,
        monthreq=10,
        curve=False,
        output=None,
        depth=0,
        month=1
    ):
        '''
        For performing a quick anomaly bar chart of specified data onto an axis
        If axis is not given, one will be generated.
        Returns figure and axis object even when input
        '''

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import datetime as dt
        import numpy as np
        import utils.betaplotutils as pu

        # Copy the pandas DF so I don't do anything to the raw data
        tmpdf = self[dkey].copy()
        # Replace the dates with just their year
        tmpdf['date'] = np.array([i.year for i in tmpdf['date']])

        # Calculate the overall average (set zero-point)
        total_avg = np.nanmean(tmpdf['value'])

        xpoints, ypoints = [[] for i in range(2)]
        for yeardftup in tmpdf.groupby('date', sort=True):
            year = yeardftup[0]
            yeardf = yeardftup[1]
            if len(yeardf['date']) < monthreq:
                continue

            tmpx = mpl.dates.date2num(
                dt.datetime(year=year, month=1, day=1)
            )
            tmpy = yeardf['value'].mean()
            if diff:
                tmpy -= total_avg

            xpoints.append(tmpx)
            ypoints.append(tmpy)

        # Make Sure Axis Exists
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)

        # Set colour based off Region, if possible
        if c is not None:
            pass
        elif self.region is not None and c is None:
            c = pu.getcolour(self.region)
        else:
            c = 'k'

        if self.region is not None:
            label = "{} Basin".format(self.region)
        else:
            label = ""

        dname = pu.get_name(dkey)
        dunit = pu.get_units(dkey)

        ylabel = "{} Yearly Anomaly ({})".format(
            dname, dunit
        ) if diff else "{} Yearly Average ({})".format(
            dname, dunit
        )

        xpoints = np.array(xpoints)
        ypoints = np.array(ypoints)

        ax.bar(
            xpoints + offset,
            ypoints,
            width=bwidth,
            color=c,
            alpha=1,
            label=label
        )

        if diff:
            ax.axhline(y=0, color='k', lw=0.8)

        if curve:
            from scipy.interpolate import interp1d
            curve_y = np.array(ypoints) * .75
            curve_x = np.arange(xpoints[0], xpoints[-1], 10)
            intf = interp1d(xpoints, curve_y, kind='quadratic')
            ax.plot(
                curve_x,
                intf(curve_x),
                '--',
                color=c,
                linewidth=2,
                alpha=0.75
            )

        cminx, cmaxx = ax.get_xlim()
        minx = min([np.floor(min(xpoints) / 365.25) * 365.25, cminx])
        maxx = max([np.ceil(max(xpoints) / 365.25) * 365.25 + 280, cmaxx])

        pu.easy_axformat(
            fig,
            ax,
            xlim=(minx, maxx),
            grid=True,
            xtime='y',
            leg=True,
            ylabel=ylabel,
        )

        if output is not None:
            import pandas as pd
            import os

            if os.path.exists(output):
                outdict = pd.read_csv(output).to_dict(orient='list')

            else:
                outdict = {
                    'Region': [],
                    'Year': [],
                    'Month': [],
                    'Depth': [],
                    'Anomaly': []
                }
            if self.region is not None:
                outdict['Region'].extend([self.region] * len(xpoints))
            else:
                outdict['Region'].extend(['Not Specified'] * len(xpoints))

            outdict['Year'].extend(
                [i.year for i in mpl.dates.num2date(xpoints)]
            )
            outdict['Month'].extend(
                [month] * len(xpoints)
            )
            outdict['Depth'].extend([0] * len(xpoints))
            outdict['Anomaly'].extend(ypoints)

            outstr = pd.DataFrame(
                data=outdict
            ).reindex(
                ['Region', 'Year', 'Month', 'Depth', 'Anomaly'], axis=1
            ).to_csv(
                index=False
            )

            with open(output, 'w+') as f:
                f.write(outstr)

        return fig, ax

    def stations(self, dkey, fig=None, ax=None, base=None):
        '''
        for plotting the points where measurements were taken.
        for MODIS data, this should return an error
        '''
        import utils.betaplotutils as pu
        import matplotlib.pyplot as plt

        # Check is it is modis data, raise error
        if 'lat' not in list(self[dkey]):
            raise RuntimeError("No location data in Dataset")

        # Make sure figure/ais exists
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)

        # Unpack lat/lon
        lons = self[dkey]['lon']
        lats = self[dkey]['lat']

        # If a basemap is needed, generate
        if base is not None:
            lake = pu.lake(lake=base, bboxpad=0.5)
            lake.plot(fig=fig, ax=ax)

        # If there is a source stored within the file, use different markers
        if 'source' not in list(self[dkey]):
            ax.scatter(
                lons,
                lats,
                marker='o',
                color='k',
                label="",
                s=24
            )
        else:
            epalon = lons[self[dkey]['source'] == 'epa']
            epalat = lats[self[dkey]['source'] == 'epa']
            ax.scatter(
                epalon,
                epalat,
                marker='^',
                color='k',
                label="EPA",
                s=24
            )
            eccclon = lons[self[dkey]['source'] == 'eccc']
            eccclat = lats[self[dkey]['source'] == 'eccc']
            ax.scatter(
                eccclon,
                eccclat,
                marker='s',
                color='k',
                label="ECCC",
                s=24
            )
            ax.legend()

        return fig, ax

    def split(self):
        '''
        Break a dataset into the three lake erie basins
        Will raise error for data with no internal coordinates
        '''
        import pandas as pd
        import numpy as np
        west = datset(None)
        cent = datset(None)
        east = datset(None)
        for dkey in self.dkeys():
            if 'lat' not in list(self[dkey]):
                raise RuntimeError("This data has no associated location")
            lons = np.array(self[dkey]['lon'])

            westinds = lons <= -82.448
            eastinds = lons >= -80.281
            centinds = (~westinds & ~eastinds)

            west[dkey] = {}
            cent[dkey] = {}
            east[dkey] = {}

            for col in list(self[dkey]):
                west[dkey][col] = self[dkey][col][westinds]
                cent[dkey][col] = self[dkey][col][centinds]
                east[dkey][col] = self[dkey][col][eastinds]

            west[dkey] = pd.DataFrame(data=west[dkey])
            cent[dkey] = pd.DataFrame(data=cent[dkey])
            east[dkey] = pd.DataFrame(data=east[dkey])

        west.region = "Western"
        cent.region = "Central"
        east.region = "Eastern"

        return west, cent, east

    def combine(self, inputs):
        if not isinstance(inputs, list):
            inputs = list(inputs)
        for d in inputs:
            self.append(None, None, newdat=d)


# ----------------------------------------------------------------------------
# READING IN DATA FILES
# ----------------------------------------------------------------------------
def beta_parsefile(filename, filetype, **kwargs):
    '''
    WORK IN PROGRESS
    All-encompassing file for reading in data
    '''
    import numpy as np
    import pandas as pd
    import utils.betaplotutils as pu
    import datetime as dt

    # Initialise dataset object
    tmpdict = datset(filetype)

    # If parsing
    if 'eccc' in filetype.lower():

        # Unpack ints and strings from file
        lat, lon, depth_from, depth_to, value = np.loadtxt(
            filename,
            skiprows=1,
            delimiter="\t",
            dtype=None,
            usecols=(6, 7, 12, 13, 17),
            unpack=True
        )
        value[value < 0] = np.nan
        # Unpack string variables
        date, tdate, abbrev, units = np.loadtxt(
            filename,
            skiprows=1,
            delimiter="\t",
            dtype=np.str,
            usecols=(1, 10, 18, 20),
            unpack=True
        )

        all_datakeys = list(set(abbrev))
        date = np.array([
            dt.datetime.strptime(i, "%Y/%m/%d") for i in date
        ])
        all_data = [
            ('date', date),
            ('tdate', tdate),
            ('lat', lat),
            ('lon', lon),
            ('depth', depth_from),
            ('value', value),
            ('units', units)
        ]

        for key in all_datakeys:
            tstkey = key.replace('"', '')
            if tstkey == "CAC":
                dkey = 'chla'
            elif tstkey in ['PH', 'I-PH']:
                dkey = 'PH'
            elif tstkey in ['I-POC', 'POC']:
                dkey = 'car'
            elif tstkey in ["D OXY P", 'D 02 W']:
                dkey = 'oxy'
            elif tstkey in ["TP-P-F", 'TP-P-UF', 'I-TF P', 'I-T P', 'TP', 'TF P']:
                dkey = 'phos'
            elif tstkey in ["CL-F", "CL F"]:
                dkey = 'clf'
            elif tstkey in ['I-N', 'N TP']:
                dkey = 'nit'
            elif tstkey in ['I-NO3+2F', 'NO3NO2 F']:
                dkey = 'nox'
            elif tstkey in ['I-NH3', 'NH3']:
                dkey = 'nh3'
            elif tstkey in ['CA-F/ICP', 'CA-F']:
                dkey = 'ca'
            elif tstkey in ['MG-FICP', 'MG-F']:
                dkey = 'mg'
            elif tstkey == 'NA-FICP':
                dkey = 'na'
            elif tstkey in ['K-FICP', 'K-F']:
                dkey = 'k'
            elif tstkey in ['AL/T-OES', 'AL/T-MS']:
                dkey = 'al'
            elif tstkey in ['FE/T-OES', 'FE/T-MS']:
                dkey = 'fe'
            else:
                continue

            keepers = abbrev == key
            keepers = keepers & (depth_to == 0)

            # If keepers is empty, there is no available data for us
            if np.sum(keepers) == 0:
                continue

            # If data for this category has already been added, different
            # Steps are needed
            if dkey in tmpdict.dkeys():
                newdict = {}
                for inputkey, inputdat in all_data[:-1]:
                    newdict[inputkey] = inputdat[keepers]
                newdict = pd.DataFrame(
                    data=newdict
                )
                tmpdict[dkey] = tmpdict[dkey].append(newdict)
            else:
                tmpdict[dkey] = {}
                for inputkey, inputdat in all_data[:-1]:
                    tmpdict[dkey][inputkey] = inputdat[keepers]
                tmpdict[dkey] = pd.DataFrame(
                    data=tmpdict[dkey]
                )
            tmpdict[dkey].units = units[keepers][0]

    elif 'modis' in filetype.lower():
        year, moday, chla, chla_cov, lwst, lwst_cov,\
            ice, ice_cov, wind_v, wind_d, cloud, precip, snow,\
            airtemp, water_lev, inflow, sus_sed, phos = np.loadtxt(
                filename,
                skiprows=1,
                unpack=True,
                delimiter="\t",
                usecols=list(range(17)) + [19]
            )
        moday = [int(i) for i in moday]
        year = [int(i) for i in year]

        if np.nanmax(moday) < 13:
            moday = [
                str(i) if i >= 10 else "0{}".format(i) for i in moday
            ]
            dates = np.array([
                dt.datetime.strptime(
                    "{}{}".format(i, j), "%Y%m"
                ) for i, j in zip(year, moday)
            ])
        else:
            moday = [
                str(i) if (len(str(i))) == 10 else "0" * (3 - len(str(i))) + str(i) for i in moday
            ]
            dates = np.array([
                dt.datetime.strptime(
                    "{}{}".format(int(i), j), "%Y%j"
                ) for i, j in zip(year, moday)
            ])
        all_data = [
            ('chla', [chla, chla_cov]),
            ('lwst', [lwst, lwst_cov]),
            ('ice', [ice, ice_cov]),
            ('wind', [wind_v, wind_d]),
            ('cloud', cloud),
            ('precip', precip),
            ('snow', snow),
            ('airtemp', airtemp),
            ('water_lev', water_lev),
            ('inflow', inflow),
            ('sus_sed', sus_sed),
            ('phos', phos)
        ]

        for dkey, data in all_data:
            units = pu.get_units(dkey)
            tmpdict[dkey] = {}
            if dkey in ['chla', 'lwst', 'ice']:
                tmpdict[dkey]['value'] = data[0]
                tmpdict[dkey]['cov'] = data[1]
            elif dkey == 'wind':
                tmpdict[dkey]['value'] = data[0]
                tmpdict[dkey]['dir'] = data[1]
            else:
                tmpdict[dkey]['value'] = data
            tmpdict[dkey]['date'] = dates
            tmpdict[dkey] = pd.DataFrame(data=tmpdict[dkey])
            tmpdict[dkey].units = units

            if 'west' in filename.lower():
                tmpdict.region = "Western"
            elif 'cent' in filename.lower():
                tmpdict.region = "Central"
            elif 'east' in filename.lower():
                tmpdict.region = "Eastern"

    elif 'epa' in filetype.lower():
        # EPA Data is all stored in separate files
        # Thusly multiple files must be read in
        import glob
        import os

        epadir = os.path.dirname(filename)
        epafiles = glob.glob(os.path.normpath("{}/*.csv".format(epadir)))
        stationfile = "F:/GlobalWaterFutures/v2/insitu/EPA/station_locations.txt"
        stations = {}

        stationsdf = pd.read_csv(stationfile)
        for row in stationsdf.itertuples():
            stations[row[1]] = [row[2], row[3]]

        for epafile in epafiles:
            # Determine dkey from filename
            if "chla" in epafile.lower():
                dkey = 'chla'
            elif "cl." in epafile.lower():
                dkey = 'clf'
            elif "no" in epafile.lower():
                dkey = 'nox'
            elif 'oxy' in epafile.lower():
                dkey = 'oxy'
            elif 'ph' in epafile.lower():
                dkey = 'PH'
            elif 't.' in epafile.lower():
                dkey = 'lwst'
            elif 'po4' in epafile.lower():
                dkey = 'phos'
            elif 'totp' in epafile.lower():
                dkey = 'topt'
            else:
                raise RuntimeError

            # Read in CSV file as dataframe
            tmp = pd.read_csv(
                epafile,
                skiprows=1,
                usecols=[
                    7, 11, 13, 16, 17, 21, 22
                ],
                names=[
                    'station',
                    'date',
                    'depth',
                    'stype',
                    'qctype',
                    'value',
                    'units'
                ],
                dtype={
                    'station': np.str,
                    'date': np.str,
                    'depth': np.float,
                    'stype': np.str,
                    'qctype': np.str,
                    'value': np.float,
                    'units': np.str
                }
            )

            # Filter out non field samples
            keepers1 = np.array(
                ['individual' in i.lower() or 'insitu' in i.lower() for i in tmp['stype']]
            )
            keepers2 = np.array(
                ['routine' in i.lower() if isinstance(i, np.str) else False for i in tmp['qctype']]
            )
            keepers = keepers1 & keepers2
            tmp = tmp[keepers]
            del keepers, keepers1, keepers2

            # Put  longitudes and latitudes in place of station
            lat, lon = zip(
                *[stations[i] if i in stations.keys() else [np.nan, np.nan] for i in tmp['station']]
            )
            tmp['lat'] = np.array(lat)
            tmp['lon'] = np.array(lon)

            # Remove stations which don't have coords
            keepers = [i in stations.keys() for i in tmp['station']]
            keepers = (
                keepers &
                np.isfinite(tmp['value']) &
                np.isfinite(tmp['depth'])
            )
            tmp = tmp[keepers]

            # Convert dates from strings to datetimes
            tmp['date'] = np.array(
                [dt.datetime.strptime(
                    i[0:10], "%Y/%m/%d"
                ) for i in tmp['date']]
            )
            dropcols = ['stype', 'units', 'qctype', 'station']
            tmp.drop(dropcols, axis=1, inplace=True)
            tmp.sort_values('date', inplace=True)
            tmpdict[dkey] = tmp.reset_index(drop=True)

    # Make sure date is the front column
    for dkey in tmpdict.keys():
        tmpdf = tmpdict[dkey]
        cols = list(tmpdf)
        cols.insert(0, cols.pop(cols.index('date')))
        if 'cov' in cols:
            cols.insert(-1, cols.pop(cols.index('cov')))
        tmpdict[dkey] = tmpdf.reindex(columns=cols)
        tmpdict[dkey].units = pu.get_units(dkey)

    return tmpdict
