"""Render "2025: The Year in Weather" as a 1080p MP4.

Run prepare_data.py first, then:  python3 make_video.py
Requires numpy, pandas, matplotlib and imageio-ffmpeg (for the ffmpeg binary).
"""
import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BUILD = os.path.join(HERE, "build")
OUT = os.path.join(HERE, "weather_2025_summary.mp4")

W, H, FPS, DPI = 1920, 1080, 30, 100

BG = "#0b1020"
PANEL = "#141b30"
FG = "#eef1f7"
MUTED = "#8b94aa"
GRID = "#222b44"
GOLD = "#ffc857"
WARM = "#ff5a3c"
COOL = "#3d9bff"

# Ed Hawkins-style warming-stripes palette
STRIPES = LinearSegmentedColormap.from_list(
    "stripes", ["#08306b", "#2171b5", "#6baed6", "#c6dbef",
                "#fee0d2", "#fc9272", "#de2d26", "#67000d"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": FG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
})

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
GIS = pd.read_csv(os.path.join(DATA, "gistemp.csv"))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ANNUAL = GIS.set_index("Year").Annual
RANK_2025 = int((ANNUAL.sort_values(ascending=False).index == 2025).argmax()) + 1
ORDINAL = {1: "warmest", 2: "2nd-warmest", 3: "3rd-warmest"}.get(RANK_2025, f"#{RANK_2025}")
MONTHLY = GIS.set_index("Year")[MONTHS]
JAN_RECORD = MONTHLY.Jan.idxmax() == 2025
MIN_2025 = MONTHLY.loc[2025].min()

CITY = pd.read_csv(os.path.join(DATA, "cities_2025.csv"), parse_dates=["date"])
CITY_NAMES = list(dict.fromkeys(CITY.city))
CITY_SERIES = {}
for name in CITY_NAMES:
    c = CITY[CITY.city == name].reset_index(drop=True)
    smooth = c.tmean.rolling(7, center=True, min_periods=1).mean().values
    CITY_SERIES[name] = dict(t=smooth, clim=c.clim.values, anom=c.anom.values,
                             tmax=c.tmax.values)
DATES = pd.date_range("2025-01-01", "2025-12-31")

EVENTS = [
    ("2025-01-07", "JAN 7", "Los Angeles wildfires", WARM,
     "The Palisades and Eaton fires killed 31 people and destroyed 16,000+ structures.",
     "About $61 billion in losses, the costliest wildfire disaster in U.S. history."),
    ("2025-06-26", "JUN – SEP", "Pakistan monsoon floods", COOL,
     "Monsoon rains flooded Khyber Pakhtunkhwa, Punjab and Sindh.",
     "More than 1,000 deaths, and 3 million+ people needed assistance."),
    ("2025-06-29", "LATE JUNE – AUGUST", "European heatwaves", WARM,
     "Temperatures reached 46°C in Spain and Portugal at the end of June.",
     "Repeated heatwaves followed through July and August."),
    ("2025-07-04", "JUL 4", "Texas Hill Country flash flood", COOL,
     "Overnight downpours sent the Guadalupe River surging within hours.",
     "At least 135 deaths, one of the deadliest U.S. inland floods in modern history."),
    ("2025-08-05", "AUG 5", "Japan's hottest day on record", WARM,
     "Isesaki, Gunma reached 41.8°C, a new national record.",
     "It beat the previous record of 41.1°C, set in 2018 and matched in 2020."),
    ("2025-08-15", "AUGUST", "Europe's worst wildfire season", WARM,
     "Over 1 million hectares burned across the EU, the most in EFFIS records (since 2006).",
     "Spain lost ~380,000 ha, its worst season since 1994."),
    ("2025-10-28", "OCT 28", "Hurricane Melissa", "#b86bff",
     "Category 5 landfall in Jamaica with 185 mph winds, the island's first Cat 5.",
     "Central pressure of 892 mb, among the most intense Atlantic landfalls ever recorded."),
    ("2025-11-27", "LATE NOVEMBER", "Cyclone Ditwah and Asian floods", COOL,
     "Sri Lanka's worst floods and landslides in decades, with over 600 deaths.",
     "Deadly flooding also struck Sumatra, southern Thailand and Malaysia."),
    ("2025-11-30", "JUN – NOV", "A strange Atlantic season", "#b86bff",
     "Three Category 5 hurricanes formed, the second-most in a single season.",
     "Still, no hurricane made U.S. landfall for the first time since 2015."),
]

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ease(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ramp(t, start, length=0.6):
    """0 -> 1 fade starting at `start` seconds."""
    return ease((t - start) / length)


def stripes_band(fig, rect, upto=1.0, alpha=1.0):
    ax = fig.add_axes(rect)
    vals = ANNUAL.values
    n = max(1, int(round(len(vals) * upto)))
    img = np.full((1, len(vals)), np.nan)
    img[0, :n] = vals[:n]
    ax.imshow(img, aspect="auto", cmap=STRIPES, vmin=-0.7, vmax=1.3,
              interpolation="nearest", alpha=alpha)
    ax.set_axis_off()
    return ax


def header(fig, title, subtitle, a=1.0):
    fig.text(0.06, 0.905, title, fontsize=44, weight="bold", alpha=a, va="center")
    fig.text(0.06, 0.848, subtitle, fontsize=22, color=MUTED, alpha=a, va="center")


def fade_overlay(fig, t, dur, fin=0.5, fout=0.5):
    env = min(ease(t / fin), ease((dur - t) / fout))
    if env < 1:
        fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                     facecolor=BG, alpha=1 - env, zorder=1000))


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=16, length=0)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# scenes: each draw(fig, t, dur)
# --------------------------------------------------------------------------


def scene_title(fig, t, dur):
    stripes_band(fig, [0, 0.0, 1, 0.16], upto=ease(t / 3.0))
    fig.text(0.5, 0.60, "2025", fontsize=210, weight="bold", ha="center", va="center",
             alpha=ramp(t, 0.2, 1.0))
    fig.text(0.5, 0.39, "THE YEAR IN WEATHER", fontsize=46, ha="center", va="center",
             color=GOLD, alpha=ramp(t, 0.9, 0.8))
    fig.text(0.5, 0.30, "Global heat, city by city, and the events that defined the year",
             fontsize=24, ha="center", color=MUTED, alpha=ramp(t, 1.5, 0.8))


def scene_global(fig, t, dur):
    header(fig, "The planet stayed near record heat",
           "Global surface temperature anomaly vs 1951–1980 average  ·  NASA GISTEMP v4",
           ramp(t, 0))
    years = ANNUAL.index.values
    vals = ANNUAL.values
    grow = ease((t - 0.6) / 7.5)
    n = int(round(len(years) * grow))
    ax = fig.add_axes([0.07, 0.12, 0.54, 0.66])
    style_axis(ax)
    cols = STRIPES(Normalize(-0.7, 1.3)(vals))
    if n:
        ax.bar(years[:n], vals[:n], width=0.82, color=cols[:n])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(1878, 2028)
    ax.set_ylim(-0.7, 1.55)
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["−0.5°C", "0", "+0.5°C", "+1.0°C", "+1.5°C"])
    ax.set_xticks([1880, 1920, 1960, 2000, 2025])

    if n:
        yr, v = years[n - 1], vals[n - 1]
        fig.text(0.665, 0.70, f"{yr}", fontsize=96, weight="bold", alpha=ramp(t, 0.6))
        fig.text(0.665, 0.62, f"{v:+.2f}°C", fontsize=48, alpha=ramp(t, 0.6),
                 color=GOLD if yr == 2025 else (WARM if v > 0.3 else COOL if v < -0.1 else FG))

    a = ramp(t, 8.6, 0.8)
    if a > 0:
        ax.annotate("2025", xy=(2025, vals[-1]), xytext=(2003, 1.42), fontsize=20,
                    color=GOLD, weight="bold", alpha=a, ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2, alpha=a))
        ax.bar(years[-1], vals[-1], width=0.9, color=GOLD, alpha=a)
        lines = [
            (8.8, f"{ORDINAL.capitalize()} in NASA's data", FG, 26, "bold"),
            (9.4, "3rd-warmest per NOAA & Copernicus,", MUTED, 22, "normal"),
            (9.4, "a near tie with 2023, behind 2024", MUTED, 22, "normal"),
            (10.2, "≈1.47°C above pre-industrial", GOLD, 26, "bold"),
            (10.2, "Copernicus ERA5, 1850–1900 baseline", MUTED, 18, "normal"),
        ]
        y = 0.50
        for start, txt, c, fs, wgt in lines:
            fig.text(0.665, y, txt, fontsize=fs, color=c, weight=wgt, alpha=ramp(t, start))
            y -= 0.055 if fs >= 26 else 0.045


def scene_monthly(fig, t, dur):
    header(fig, "2025 month by month, vs 2023 and 2024",
           "Global monthly temperature anomaly vs 1951–1980  ·  NASA GISTEMP v4", ramp(t, 0))
    ax = fig.add_axes([0.07, 0.14, 0.62, 0.62])
    style_axis(ax)
    x = np.arange(12)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(0.8, 1.6)
    ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_yticklabels([f"+{v:.1f}°C" for v in [0.8, 1.0, 1.2, 1.4, 1.6]])
    for yr, c, ls in [(2023, "#c07a5a", "--"), (2024, "#d9534f", "-")]:
        a = ramp(t, 0.5 if yr == 2023 else 1.0)
        ax.plot(x, MONTHLY.loc[yr].values, color=c, lw=3, ls=ls, alpha=0.75 * a)
        ax.text(11.25, MONTHLY.loc[yr].values[-1], f" {yr}", color=c, fontsize=18,
                va="center", alpha=a, weight="bold")
    v25 = MONTHLY.loc[2025].values
    prog = ease((t - 1.6) / 5.0) * 11
    k = int(prog)
    frac = prog - k
    xs = list(x[:k + 1])
    ys = list(v25[:k + 1])
    if k < 11:
        xs.append(k + frac)
        ys.append(v25[k] + (v25[k + 1] - v25[k]) * frac)
    if t > 1.6:
        ax.plot(xs, ys, color=GOLD, lw=5, solid_capstyle="round")
        ax.scatter(x[:k + 1], v25[:k + 1], color=GOLD, s=60, zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], color=GOLD, s=260, zorder=6, edgecolor=BG, lw=3)
        if k >= 11:
            ax.text(11.25, v25[-1] - 0.03, " 2025", color=GOLD, fontsize=18,
                    va="center", weight="bold")
        idx = min(11, int(round(xs[-1])))
        fig.text(0.73, 0.70, MONTHS[idx] + " 2025", fontsize=46, weight="bold",
                 alpha=ramp(t, 1.6))
        fig.text(0.73, 0.63, f"{v25[idx]:+.2f}°C", fontsize=40, color=GOLD,
                 alpha=ramp(t, 1.6))
    notes = []
    if JAN_RECORD:
        notes.append("January 2025 was the warmest January on record, even with La Niña in the Pacific")
    notes.append(f"No month in 2025 dropped below +{math.floor(MIN_2025 * 10) / 10:.1f}°C")
    notes.append("The year eased off 2024's pace but stayed far above anything before 2023")
    y = 0.50
    for i, n in enumerate(notes):
        a = ramp(t, 6.9 + 0.5 * i)
        fig.text(0.73, y, "•", fontsize=24, color=GOLD, alpha=a, va="top")
        fig.text(0.748, y, wrap(n, 30), fontsize=20, color=FG, alpha=a, va="top",
                 linespacing=1.35)
        y -= 0.13


def wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def scene_cities(fig, t, dur):
    header(fig, "How 2025 felt in 12 cities",
           "7-day mean temperature: red = warmer, blue = cooler than the 1991–2020 "
           "normal (grey)  ·  Meteostat", ramp(t, 0))
    sweep = ease((t - 0.8) / 11.0)
    day = int(round(sweep * 364))
    cols, rows = 4, 3
    x0, y0, cw, ch, gx, gy = 0.05, 0.08, 0.212, 0.215, 0.024, 0.045
    hold = t > 12.2
    finals = {n: CITY_SERIES[n]["anom"].mean() for n in CITY_NAMES}
    top = max(finals, key=finals.get)
    for i, name in enumerate(CITY_NAMES):
        r, c = divmod(i, cols)
        a = ramp(t, 0.2 + 0.05 * i, 0.5)
        rect = [x0 + c * (cw + gx), y0 + (rows - 1 - r) * (ch + gy), cw, ch]
        s = CITY_SERIES[name]
        ax = fig.add_axes(rect)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        lo = min(s["t"].min(), s["clim"].min())
        hi = max(s["t"].max(), s["clim"].max())
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad * 2.4)
        ax.set_xlim(0, 364)
        xx = np.arange(365)
        ax.plot(xx, s["clim"], color=MUTED, lw=1.5, alpha=0.7 * a)
        if day > 0:
            sl = slice(0, day + 1)
            ax.fill_between(xx[sl], s["clim"][sl], s["t"][sl],
                            where=s["t"][sl] >= s["clim"][sl], color=WARM,
                            alpha=0.85 * a, interpolate=True, lw=0)
            ax.fill_between(xx[sl], s["clim"][sl], s["t"][sl],
                            where=s["t"][sl] < s["clim"][sl], color=COOL,
                            alpha=0.85 * a, interpolate=True, lw=0)
            ax.plot(xx[sl], s["t"][sl], color=FG, lw=1.2, alpha=a)
            ax.axvline(day, color=FG, lw=1, alpha=0.35 * a * (1 - ramp(t, 11.8)))
        ytd = round(s["anom"][:day + 1].mean(), 1) + 0.0
        edge = GOLD if (hold and name == top) else None
        if edge:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                   edgecolor=GOLD, lw=4 * ramp(t, 12.2), zorder=10))
        ax.text(0.04, 0.92, name, transform=ax.transAxes, fontsize=20, weight="bold",
                va="top", alpha=a)
        ax.text(0.96, 0.92, f"{ytd:+.1f}°C", transform=ax.transAxes, fontsize=20,
                weight="bold", va="top", ha="right", alpha=a,
                color=FG if ytd == 0 else WARM if ytd > 0 else COOL)
    if t > 0.8:
        label = DATES[day].strftime("%B %-d") if not hold else "Full-year average vs normal"
        fig.text(0.94, 0.905, label, fontsize=30, weight="bold", ha="right", va="center",
                 color=GOLD)
    if hold:
        warm_n = sum(v > 0.05 for v in finals.values())
        fig.text(0.5, 0.025,
                 f"{warm_n} of 12 cities ran warmer than normal · {top} led at "
                 f"{finals[top]:+.1f}°C above its 1991–2020 average",
                 fontsize=22, ha="center", color=FG, alpha=ramp(t, 12.4))


EVENT_SLOT = 2.9
EVENT_T0 = 1.4


def scene_events(fig, t, dur):
    header(fig, "The events that defined 2025", "Selected extreme weather disasters of the year",
           ramp(t, 0))
    # timeline
    ax = fig.add_axes([0.06, 0.66, 0.88, 0.1])
    ax.set_xlim(0, 365)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.plot([0, 365], [0, 0], color=GRID, lw=4, solid_capstyle="round")
    starts = pd.date_range("2025-01-01", periods=12, freq="MS")
    for i, m in enumerate(starts):
        d = m.dayofyear - 1
        ax.plot([d, d], [-0.15, 0.15], color=MUTED, lw=1.5)
        ax.text(d + 15, -0.6, MONTHS[i].upper(), ha="center", fontsize=15, color=MUTED)
    cur = int((t - EVENT_T0) // EVENT_SLOT)
    for i, ev in enumerate(EVENTS):
        start = EVENT_T0 + i * EVENT_SLOT
        if t < start:
            continue
        d = pd.Timestamp(ev[0]).dayofyear - 1
        color = ev[3]
        active = i == min(cur, len(EVENTS) - 1)
        pop = ramp(t, start, 0.35)
        size = (420 if active else 140) * pop
        ax.scatter([d], [0], s=size, color=color, zorder=5, edgecolor=BG, lw=2)
        if active:
            ax.scatter([d], [0], s=size * 3, color=color, alpha=0.2, zorder=4)
    # card for the active event
    if t >= EVENT_T0:
        i = min(cur, len(EVENTS) - 1)
        start = EVENT_T0 + i * EVENT_SLOT
        last = i == len(EVENTS) - 1
        a = ramp(t, start, 0.35) * (1.0 if last else (1 - ramp(t, start + EVENT_SLOT - 0.3, 0.3)))
        date, when, title, color, l1, l2 = EVENTS[i]
        slide = (1 - ramp(t, start, 0.45)) * 0.02
        fig.patches.append(FancyBboxPatch(
            (0.10, 0.12 - slide), 0.80, 0.42, transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.015", facecolor=PANEL,
            edgecolor=color, lw=3, alpha=a))
        fig.text(0.145, 0.47 - slide, when, fontsize=24, color=color, weight="bold", alpha=a)
        fig.text(0.145, 0.375 - slide, title, fontsize=54, weight="bold", alpha=a)
        fig.text(0.145, 0.27 - slide, l1, fontsize=24, color=FG, alpha=a)
        fig.text(0.145, 0.20 - slide, l2, fontsize=24, color=MUTED, alpha=a)
        fig.text(0.88, 0.47 - slide, f"{i + 1} / {len(EVENTS)}", fontsize=18,
                 color=MUTED, ha="right", alpha=a)


def scene_close(fig, t, dur):
    stripes_band(fig, [0, 0.0, 1, 0.16], alpha=ramp(t, 0, 1.0))
    fig.text(0.5, 0.74, "2023 · 2024 · 2025", fontsize=80, weight="bold", ha="center",
             alpha=ramp(t, 0.2, 0.8))
    fig.text(0.5, 0.60,
             "The first three-year stretch to average more than 1.5°C\n"
             "above pre-industrial levels",
             fontsize=34, ha="center", va="center", color=GOLD, alpha=ramp(t, 0.9, 0.8),
             linespacing=1.4)
    fig.text(0.5, 0.47, "Copernicus Climate Change Service", fontsize=20, ha="center",
             color=MUTED, alpha=ramp(t, 1.2))
    fig.text(0.5, 0.33,
             "Data: NASA GISTEMP v4 · Meteostat daily station records · Copernicus C3S · "
             "NOAA NCEI & NHC · EU JRC/EFFIS · WMO · JMA",
             fontsize=17, ha="center", color=MUTED, alpha=ramp(t, 2.0))
    fig.text(0.5, 0.28,
             "City anomalies use (Tmax+Tmin)/2 against each station's 1991–2020 daily normal",
             fontsize=15, ha="center", color=MUTED, alpha=ramp(t, 2.2))
    fig.text(0.5, 0.215, "Stripes: global annual temperature, 1880–2025",
             fontsize=15, ha="center", color=MUTED, alpha=ramp(t, 2.4))


SCENES = [
    (scene_title, 5.0),
    (scene_global, 13.5),
    (scene_monthly, 11.0),
    (scene_cities, 17.0),
    (scene_events, EVENT_T0 + EVENT_SLOT * len(EVENTS) + 1.2),
    (scene_close, 8.0),
]


def frame_plan():
    plan = []
    for si, (_, dur) in enumerate(SCENES):
        n = int(round(dur * FPS))
        plan += [(si, f / FPS) for f in range(n)]
    return plan


def render_frame(fig, si, t):
    fig.clf()
    fig.patches.clear()
    draw, dur = SCENES[si]
    draw(fig, t, dur)
    fade_overlay(fig, t, dur)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()


def render_chunk(args):
    idx, frames = args
    path = os.path.join(BUILD, f"part_{idx:02d}.mp4")
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for si, t in frames:
        proc.stdin.write(render_frame(fig, si, t))
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# soundtrack: a soft synthesized ambient pad
# --------------------------------------------------------------------------


def make_music(seconds, path, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    # A minor -> F -> C -> G, 8 s per chord
    chords = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 55, 60, 64], [55, 59, 62, 67]]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)  # noqa: E731
    out = np.zeros_like(t)
    span = 8.0
    for k in range(int(seconds // span) + 1):
        chord = chords[k % len(chords)]
        t0 = k * span
        seg = (t >= t0 - 1) & (t < t0 + span + 3)
        tt = t[seg] - t0
        env = np.clip((tt + 1) / 3, 0, 1) * np.clip((span + 3 - tt) / 4, 0, 1)
        for m in chord + [chord[0] - 12]:
            f = hz(m)
            for h, amp in [(1, 1.0), (2, 0.25), (3, 0.08)]:
                detune = 1 + 0.0015 * math.sin(k + m)
                out[seg] += amp * env * np.sin(2 * math.pi * f * h * detune * tt + m)
    # gentle slow tremolo and a feedback delay for space
    out *= 0.85 + 0.15 * np.sin(2 * math.pi * 0.12 * t)
    d = int(0.37 * sr)
    for _ in range(3):
        out[d:] += 0.35 * out[:-d]
    out /= np.abs(out).max()
    fade = np.minimum(1, np.minimum(t / 2.0, (seconds - t) / 3.0))
    out *= 0.22 * np.clip(fade, 0, 1)
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(BUILD, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--still":
        # render single frames for previewing: --still <scene> <t>
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        si, t = int(sys.argv[2]), float(sys.argv[3])
        raw = render_frame(fig, si, t)
        img = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        plt.imsave(os.path.join(BUILD, f"still_{si}_{t:.1f}.png"), img)
        return
    plan = frame_plan()
    workers = os.cpu_count() or 2
    size = math.ceil(len(plan) / workers)
    chunks = [(i, plan[i * size:(i + 1) * size]) for i in range(workers)]
    print(f"rendering {len(plan)} frames ({len(plan) / FPS:.1f}s) on {workers} workers")
    with Pool(workers) as pool:
        parts = pool.map(render_chunk, chunks)
    listing = os.path.join(BUILD, "parts.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    music = os.path.join(BUILD, "music.wav")
    make_music(len(plan) / FPS, music)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-i", music, "-c:v", "copy", "-c:a", "aac", "-b:a",
                    "128k", "-shortest", "-movflags", "+faststart", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
