"""Download and pre-process the data behind the "2025 in Weather" video.

Sources
  * NASA GISTEMP v4 global land+ocean temperature anomalies (vs 1951-1980)
  * Meteostat bulk daily station data (derived from NOAA ISD / national services)

Outputs (in ./data)
  * gistemp.csv       - annual + monthly global anomalies, 1880-2025
  * cities_2025.csv   - daily 2025 mean temperature and anomaly vs 1991-2020
"""
import gzip
import io
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
METEOSTAT_URL = "https://bulk.meteostat.net/v2/daily/{}.csv.gz"
METEOSTAT_COLS = "date,tavg,tmin,tmax,prcp,snow,wdir,wspd,wpgt,pres,tsun".split(",")

# (display name, Meteostat / WMO station id), ordered roughly west -> east
CITIES = [
    ("Toronto", "71624"),
    ("New York", "72503"),
    ("London", "03772"),
    ("Madrid", "08221"),
    ("Lagos", "65201"),
    ("Cape Town", "68816"),
    ("Cairo", "62366"),
    ("Moscow", "27612"),
    ("New Delhi", "42182"),
    ("Beijing", "54511"),
    ("Tokyo", "47662"),
    ("Sydney", "94768"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "weather-2025-video"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def prepare_gistemp():
    raw = fetch(GISTEMP_URL).decode()
    df = pd.read_csv(io.StringIO(raw), skiprows=1, na_values="***")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df = df[["Year"] + months + ["J-D"]].rename(columns={"J-D": "Annual"})
    df = df[df.Year <= 2025]
    df.to_csv(os.path.join(DATA, "gistemp.csv"), index=False)
    print("gistemp:", len(df), "years; 2025 annual =", df.Annual.iloc[-1])


def prepare_cities():
    rows = []
    for name, sid in CITIES:
        raw = gzip.decompress(fetch(METEOSTAT_URL.format(sid)))
        d = pd.read_csv(io.BytesIO(raw), names=METEOSTAT_COLS, parse_dates=["date"])
        # (Tmax+Tmin)/2 is available for every station and baseline year,
        # so use it consistently rather than mixing in 24h averages.
        d["tmean"] = (d.tmax + d.tmin) / 2
        d["doy"] = d.date.dt.dayofyear.clip(upper=365)

        base = d[(d.date.dt.year >= 1991) & (d.date.dt.year <= 2020)]
        clim = base.groupby("doy").tmean.mean().reindex(range(1, 366))
        clim = clim.interpolate(limit_direction="both")
        # smooth the daily climatology with a circular 31-day window
        padded = np.concatenate([clim.values[-15:], clim.values, clim.values[:15]])
        smooth = np.convolve(padded, np.ones(31) / 31, mode="valid")
        clim = pd.Series(smooth, index=range(1, 366))

        y = d[d.date.dt.year == 2025].set_index("date")
        y = y.reindex(pd.date_range("2025-01-01", "2025-12-31"))
        y["tmean"] = y.tmean.interpolate(limit_direction="both")
        y["tmax"] = y.tmax.interpolate(limit_direction="both")
        doy = np.minimum(y.index.dayofyear, 365)
        y["clim"] = clim.loc[doy].values
        y["anom"] = y.tmean - y.clim
        base_years = base.date.dt.year.nunique()
        print(f"{name:10s} 2025 mean {y.tmean.mean():5.1f}°C  anomaly "
              f"{y.anom.mean():+.2f}°C  (baseline years: {base_years})")
        for date, r in y.iterrows():
            rows.append(dict(city=name, date=date.date(), tmean=round(r.tmean, 2),
                             tmax=r.tmax, clim=round(r.clim, 2), anom=round(r.anom, 2)))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "cities_2025.csv"), index=False)


if __name__ == "__main__":
    prepare_gistemp()
    prepare_cities()
