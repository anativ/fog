"""Render "2025" -- a movie-trailer cut of the year in weather.

Run prepare_data.py first, then:  python3 make_trailer.py
Preview a frame:                  python3 make_trailer.py --still <seconds>
Requires numpy, pandas, pillow, scipy, matplotlib (colormap only) and imageio-ffmpeg.
Fonts (SIL Open Font License) live in ./fonts.
"""
import math
import os
import subprocess
import sys
import wave
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import signal

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FONTS = os.path.join(HERE, "fonts")
BUILD = os.path.join(HERE, "build")
OUT = os.path.join(HERE, "weather_2025_trailer.mp4")

W, H, FPS = 1920, 1080, 30
BW, BH = W // 2, H // 2          # backdrops are rendered at half resolution
BAR = 140                         # letterbox bars -> 2.4:1 picture
SR = 44100

ANTON = os.path.join(FONTS, "Anton-Regular.ttf")
BEBAS = os.path.join(FONTS, "BebasNeue-Regular.ttf")
CINZEL = os.path.join(FONTS, "Cinzel-Bold.ttf")

WHITE = (1.0, 0.97, 0.93)
EMBER = (1.0, 0.55, 0.18)
HOT = (1.0, 0.32, 0.18)
ICE = (0.55, 0.78, 1.0)
GREY = (0.62, 0.62, 0.66)

STRIPES = LinearSegmentedColormap.from_list(
    "stripes", ["#08306b", "#2171b5", "#6baed6", "#c6dbef",
                "#fee0d2", "#fc9272", "#de2d26", "#67000d"])

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
GIS = pd.read_csv(os.path.join(DATA, "gistemp.csv"))
ANNUAL = GIS.Annual.values
YEARS = GIS.Year.values
CITY = pd.read_csv(os.path.join(DATA, "cities_2025.csv"))
CITY_ANOM = CITY.groupby("city", sort=False).anom.mean()
CITY_DEV = {}
for _name, _g in CITY.groupby("city", sort=False):
    _t = _g.tmean.rolling(7, center=True, min_periods=1).mean().values
    CITY_DEV[_name] = _t - _g.clim.values
WARM_CITIES = [c for c in CITY_ANOM.sort_values(ascending=False).index
               if round(CITY_ANOM[c], 1) > 0][:10]

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
RNG = np.random.default_rng(2025)


def clamp(x, lo=0.0, hi=1.0):
    return np.minimum(hi, np.maximum(lo, x))


def ease(x):
    x = float(clamp(x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def noise2d(h, w, scale, rng, octaves=4):
    """Cheap fractal value noise in [0, 1] built from upsampled random grids."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
        grid = (rng.random((gh, gw)) * 255).astype(np.uint8)
        img = Image.fromarray(grid).resize((w, h), Image.BICUBIC)
        out += amp * np.asarray(img, np.float32) / 255
        total += amp
        amp *= 0.5
        scale = max(1, scale // 2)
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def lut(stops):
    xs = np.linspace(0, 1, 256)
    pos = [s[0] for s in stops]
    return np.stack([np.interp(xs, pos, [s[1][c] for s in stops]) for c in range(3)],
                    axis=1).astype(np.float32)


FIRE_LUT = lut([(0, (0, 0, 0)), (0.3, (0.3, 0.02, 0.0)), (0.55, (0.75, 0.16, 0.02)),
                (0.8, (1.0, 0.5, 0.08)), (1.0, (1.0, 0.88, 0.55))])


def apply_lut(x, table):
    idx = (clamp(x) * 255).astype(np.uint8)
    return table[idx]


YY, XX = np.mgrid[0:BH, 0:BW].astype(np.float32)
YN, XN = YY / BH, XX / BW

# --------------------------------------------------------------------------
# precomputed textures
# --------------------------------------------------------------------------
FIRE_TEX = noise2d(BH * 3, BW, 48, RNG, 5)
FIRE_TEX2 = noise2d(BH * 3, BW, 20, RNG, 3)
CLOUD_TEX = noise2d(BH, BW * 3, 90, RNG, 5)


def _rain_tile():
    img = Image.new("L", (BW, BH * 2), 0)
    d = ImageDraw.Draw(img)
    for _ in range(1400):
        x, y = RNG.integers(0, BW), RNG.integers(0, BH * 2)
        ln = RNG.integers(14, 38)
        d.line([(x, y), (x - ln * 0.28, y + ln)], fill=int(RNG.integers(50, 170)), width=1)
    return np.asarray(img, np.float32) / 255


RAIN_TEX = _rain_tile()


def _hurricane_tex(size=1100):
    g = np.linspace(-1, 1, size, dtype=np.float32)
    x, y = np.meshgrid(g, g)
    r = np.sqrt(x * x + y * y) + 1e-4
    th = np.arctan2(y, x)
    arms = 0.5 + 0.5 * np.cos(3 * th + 8.5 * np.log(r + 0.03))
    n = noise2d(size, size, 60, RNG, 5)
    dens = (arms ** 1.4 * 0.75 + n * 0.55) * np.exp(-r * 1.25)
    dens *= smoothstep(0.035, 0.085, r)
    dens += 0.9 * np.exp(-((r - 0.075) / 0.022) ** 2)
    dens = clamp(dens * 1.25 - 0.08)
    return Image.fromarray((dens * 255).astype(np.uint8))


HURRICANE_TEX = _hurricane_tex()


def _heat_base():
    sky = np.zeros((BH, BW, 3), np.float32)
    horizon = 0.70
    gy = clamp(YN / horizon)
    for c, (top, bot) in enumerate([(0.18, 1.0), (0.02, 0.42), (0.02, 0.08)]):
        sky[..., c] = top + (bot - top) * gy ** 1.8
    cx, cy = 0.84 * BW, 0.62 * BH
    d = np.sqrt((XX - cx) ** 2 + ((YY - cy) * 1.0) ** 2)
    glow = np.exp(-d / 120)[..., None] * np.array([1.0, 0.55, 0.2], np.float32)
    core = smoothstep(62, 54, d)[..., None] * np.array([1.0, 0.95, 0.75], np.float32)
    sky = clamp(sky + glow * 0.9 + core)
    hills = horizon * BH + 10 * np.sin(np.linspace(0, 9, BW)) + 18 * noise2d(1, BW, 80, RNG, 3)[0]
    ground = YY > hills[None, :]
    sky[ground] = np.array([0.03, 0.01, 0.01], np.float32)
    return sky


HEAT_BASE = _heat_base()

EMBERS = dict(x=RNG.random(170), y=RNG.random(170), v=0.05 + 0.13 * RNG.random(170),
              ph=RNG.random(170) * 6.28, s=0.5 + RNG.random(170))


def _vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2 * 0.9)) ** 2)
    return (1 - 0.55 * smoothstep(0.55, 1.35, d))[..., None].astype(np.float32)


VIGNETTE = _vignette()
# half-resolution grain reads as film grain and keeps the encode a sane size
GRAIN = [np.asarray(Image.fromarray(RNG.normal(0, 0.018, (BH, BW)).astype(np.float32))
                    .resize((W, H), Image.BILINEAR))[..., None] for _ in range(6)]

# --------------------------------------------------------------------------
# backdrops (half resolution, float RGB)
# --------------------------------------------------------------------------


def bg_black(t, p):
    return np.zeros((BH, BW, 3), np.float32)


def bg_fire(t, p):
    o1 = int(t * 150) % FIRE_TEX.shape[0]
    o2 = int(t * 260) % FIRE_TEX2.shape[0]
    rows1 = (np.arange(BH) + o1) % FIRE_TEX.shape[0]
    rows2 = (np.arange(BH) + o2) % FIRE_TEX2.shape[0]
    n = FIRE_TEX[rows1] * 0.65 + FIRE_TEX2[rows2] * 0.35
    mask = YN ** 1.35
    inten = clamp(n * 1.55 * mask + mask * 0.35 - 0.38)
    flick = 0.9 + 0.1 * math.sin(t * 13) * math.sin(t * 7.3)
    img = apply_lut(inten * flick * p.get("gain", 1.0), FIRE_LUT)
    smoke = (CLOUD_TEX[:, (np.arange(BW) + int(t * 20)) % CLOUD_TEX.shape[1]] * 0.12)
    return clamp(img + smoke[..., None] * np.array([0.4, 0.2, 0.15], np.float32))


def bg_heat(t, p):
    amp = 3.5 * smoothstep(0.30, 0.72, YN[:, :1])
    shift = (amp * np.sin(YY[:, :1] * 0.35 + t * 8.0)).astype(np.int32)
    cols = (np.arange(BW)[None, :] + shift) % BW
    img = np.take_along_axis(HEAT_BASE, cols[..., None].repeat(3, axis=2), axis=1)
    return img * (0.9 + 0.08 * math.sin(t * 2.1))


def bg_rain(t, p):
    ox = int(t * 25) % CLOUD_TEX.shape[1]
    cloud = CLOUD_TEX[:, (np.arange(BW) + ox) % CLOUD_TEX.shape[1]]
    base = (0.03 + 0.10 * cloud * (1.2 - YN))[..., None] * np.array([0.6, 0.8, 1.1], np.float32)
    oy = int(t * 1250) % RAIN_TEX.shape[0]
    rain = RAIN_TEX[(np.arange(BH) - oy) % RAIN_TEX.shape[0]]
    img = base + rain[..., None] * np.array([0.35, 0.42, 0.5], np.float32)
    for lt in p.get("lightning", []):
        dt = t - lt
        if 0 <= dt < 0.6:
            f = math.exp(-dt / 0.08) + 0.5 * math.exp(-max(0, dt - 0.15) / 0.06) * (dt > 0.15)
            img = img + f * (0.25 + 0.6 * cloud[..., None]) * np.array([0.8, 0.85, 1.0], np.float32)
    return clamp(img)


def bg_hurricane(t, p):
    rot = HURRICANE_TEX.rotate(t * 18, resample=Image.BILINEAR)
    s = HURRICANE_TEX.size[0]
    zoom = 1.0 + 0.10 * t / 4.5
    cw, ch = int(BW / zoom), int(BH / zoom)
    x0, y0 = (s - cw) // 2, (s - ch) // 2
    crop = rot.crop((x0, y0, x0 + cw, y0 + ch)).resize((BW, BH), Image.BILINEAR)
    d = np.asarray(crop, np.float32)[..., None] / 255
    ocean = np.array([0.01, 0.04, 0.10], np.float32)
    cloud = np.array([0.85, 0.9, 1.0], np.float32)
    return ocean * (1 - d) + cloud * d * 0.85


def bg_stripes(t, p):
    dur = p["dur"]
    vis = 70
    start = (len(ANNUAL) - vis) * ease(t / dur)
    idx = np.clip((start + XN[0] * vis).astype(int), 0, len(ANNUAL) - 1)
    cols = STRIPES((ANNUAL[idx] + 0.7) / 2.0)[:, :3].astype(np.float32)
    img = np.repeat(cols[None], BH, axis=0)
    return img * (0.55 + 0.25 * (1 - np.abs(YN - 0.5) * 2))[..., None]


def bg_chart(t, p):
    img = Image.new("RGB", (BW, BH), (0, 0, 0))
    d = ImageDraw.Draw(img)
    prog = ease((t - 0.2) / (p["dur"] - 1.2))
    n = max(2, int(len(ANNUAL[:-1]) * prog))
    x = 60 + (YEARS[:n] - 1880) / 145 * (BW - 120)
    y = BH * 0.72 - ANNUAL[:n] * 190
    d.line([(60, BH * 0.72), (BW - 60, BH * 0.72)], fill=(40, 40, 48), width=1)
    for i in range(n - 1):
        c = STRIPES((ANNUAL[i] + 0.7) / 2.0)
        col = tuple(int(255 * v) for v in c[:3])
        d.line([(x[i], y[i]), (x[i + 1], y[i + 1])], fill=col, width=3)
    arr = np.asarray(img, np.float32) / 255
    glow = np.asarray(img.filter(ImageFilter.GaussianBlur(9)), np.float32) / 255
    tip = np.exp(-((XX - x[-1]) ** 2 + (YY - y[-1]) ** 2) / 90)[..., None]
    return clamp(arr + glow * 2.2 + tip * np.array([1, 0.6, 0.4], np.float32))


def bg_city(t, p):
    dev = CITY_DEV[p["city"]]
    img = Image.new("RGB", (BW, BH), (0, 0, 0))
    d = ImageDraw.Draw(img)
    mid = BH * 0.5
    k = BH * 0.035
    step = BW / len(dev)
    for i, v in enumerate(dev):
        col = (255, 80, 50) if v >= 0 else (70, 150, 255)
        x = i * step
        d.rectangle([x, min(mid, mid - v * k), x + step, max(mid, mid - v * k)], fill=col)
    arr = np.asarray(img, np.float32) / 255
    glow = np.asarray(img.filter(ImageFilter.GaussianBlur(12)), np.float32) / 255
    return clamp(arr * 0.35 + glow * 0.8) * 0.6


def bg_title(t, p):
    img = bg_fire(t, {"gain": 0.55})
    cx, cy = BW / 2, BH * 0.5
    r = np.sqrt((XX - cx) ** 2 + ((YY - cy) * 1.6) ** 2)
    glow = np.exp(-r / 260)[..., None] * np.array([0.5, 0.14, 0.03], np.float32)
    return clamp(img * 0.7 + glow * (0.8 + 0.2 * math.sin(t * 2)))


BACKDROPS = dict(black=bg_black, fire=bg_fire, heat=bg_heat, rain=bg_rain,
                 hurricane=bg_hurricane, stripes=bg_stripes, chart=bg_chart,
                 city=bg_city, title=bg_title)

# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=512)
def text_mask(s, path, size, tracking):
    f = font(path, size)
    widths = [f.getlength(ch) for ch in s]
    tw = int(sum(widths) + tracking * (len(s) - 1)) + 1
    asc, desc = f.getmetrics()
    pad = max(8, size // 4)
    img = Image.new("L", (tw + 2 * pad, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    x = pad
    for ch, w in zip(s, widths):
        d.text((x, pad), ch, font=f, fill=255)
        x += w + tracking
    return img


@lru_cache(maxsize=512)
def glow_mask(s, path, size, tracking, radius):
    return text_mask(s, path, size, tracking).filter(ImageFilter.GaussianBlur(radius))


def composite(frame, mask_img, cx, cy, color, alpha, scale=1.0):
    if alpha <= 0.003:
        return
    m = mask_img
    if abs(scale - 1.0) > 1e-3:
        m = m.resize((max(1, int(m.width * scale)), max(1, int(m.height * scale))),
                     Image.BILINEAR)
    a = np.asarray(m, np.float32)[..., None] / 255 * alpha
    x0, y0 = int(cx - m.width / 2), int(cy - m.height / 2)
    x1, y1 = x0 + m.width, y0 + m.height
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    a = a[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    region = frame[y0:y1, x0:x1]
    region *= 1 - a
    region += a * np.asarray(color, np.float32)


class Cue:
    """One line of on-screen text inside a shot (times are shot-relative)."""

    def __init__(self, s, t_in, t_out=None, face=BEBAS, size=80, y=540, x=W / 2,
                 color=WHITE, style="fade", tracking=0, glow=None, hit=False, big=False,
                 soft=False):
        self.s, self.t_in, self.t_out = s, t_in, t_out
        self.face, self.size, self.y, self.x = face, size, y, x
        self.color, self.style, self.tracking = color, style, tracking
        self.glow, self.hit, self.big, self.soft = glow, hit, big, soft

    def draw(self, frame, t, shot_dur):
        tau = t - self.t_in
        if tau < 0:
            return
        t_out = self.t_out if self.t_out is not None else shot_dur
        out = 1 - ease((t - t_out) / 0.25) if t > t_out else 1.0
        if self.style == "slam":
            k = min(1.0, tau / 0.11)
            scale = (1.45 - 0.45 * ease(k)) if k < 1 else 1 + 0.025 * (tau - 0.11)
            alpha = k * out
        elif self.style == "track":
            scale, alpha = 1.0, ease(tau / 0.9) * out
        else:
            scale, alpha = 1 + 0.018 * tau, ease(tau / 0.45) * out
        tracking = self.tracking
        if self.style == "track":
            tracking = int(self.tracking + 5 * tau)
        m = text_mask(self.s, self.face, self.size, tracking)
        if self.glow:
            g = glow_mask(self.s, self.face, self.size, tracking, self.glow[1])
            for _ in range(self.glow[2]):
                composite(frame, g, self.x, self.y, self.glow[0], alpha * 0.9, scale)
        composite(frame, m, self.x, self.y, self.color, alpha, scale)


# --------------------------------------------------------------------------
# the cut
# --------------------------------------------------------------------------
TAG = dict(face=BEBAS, size=46, tracking=12, color=(0.92, 0.85, 0.75))
SUB = dict(face=BEBAS, size=60, tracking=6)


def slam(s, t, size=230, y=520, color=WHITE, glow=(EMBER, 22, 1), **kw):
    return Cue(s, t, face=ANTON, size=size, y=y, color=color, style="slam",
               glow=glow, hit=True, **kw)


class Shot:
    def __init__(self, start, dur, bg, cues=(), push=0.06, fin=0.08, fout=0.12,
                 **params):
        self.start, self.dur, self.bg = start, dur, bg
        self.cues, self.push, self.fin, self.fout = list(cues), push, fin, fout
        self.params = dict(params, dur=dur)


def event_shot(start, dur, bg, tag, big, sub, glow_col=EMBER, size=230, **params):
    return Shot(start, dur, bg, [
        Cue(tag, 0.15, y=330, **TAG),
        slam(big, 0.75, size=size, glow=(glow_col, 24, 1)),
        Cue(sub, 1.25, y=720, **SUB),
    ], **params)


def build_shots():
    shots = [
        Shot(0.0, 4.0, "black", [
            Cue("THE FOLLOWING IS BASED ON", 0.5, 3.3, face=CINZEL, size=34, y=505,
                style="track", tracking=10, color=GREY),
            Cue("REAL WEATHER DATA", 0.9, 3.3, face=CINZEL, size=34, y=565,
                style="track", tracking=10, color=GREY),
        ], fin=0.01, fout=0.3),
        Shot(4.0, 4.6, "stripes", [
            Cue("FOR 146 YEARS", 0.5, 2.1, size=110, tracking=18, y=540),
            Cue("WE'VE MEASURED EVERY DEGREE", 2.3, 4.2, size=96, tracking=12, y=540),
        ], fin=0.6, fout=0.3, push=0.05),
        Shot(8.6, 4.4, "chart", [
            Cue("THEN CAME", 0.6, 1.9, size=110, tracking=18, y=300),
            Cue("THE HOTTEST YEARS EVER RECORDED", 2.0, 4.1, size=84, tracking=10, y=300,
                glow=(HOT, 14, 1)),
        ], fin=0.3, push=0.03),
        Shot(13.0, 1.8, "black", [
            slam("AND IT WASN'T OVER.", 0.0, size=120, glow=(HOT, 18, 1)),
        ], fin=0.0),
        event_shot(14.8, 3.6, "fire", "JANUARY  ·  LOS ANGELES", "16,000",
                   "HOMES AND BUSINESSES DESTROYED"),
        event_shot(18.4, 3.4, "rain", "JUNE – SEPTEMBER  ·  PAKISTAN", "3 MILLION+",
                   "PEOPLE IN NEED OF AID AFTER MONSOON FLOODS", glow_col=ICE),
        event_shot(21.8, 3.2, "heat", "LATE JUNE  ·  SPAIN & PORTUGAL", "46°C",
                   "EUROPE'S SUMMER OF FIRE AND HEAT", size=260),
        event_shot(25.0, 3.4, "rain", "JULY 4  ·  TEXAS HILL COUNTRY", "20 INCHES",
                   "OF RAIN. A RIVER RISING BY THE MINUTE.", glow_col=ICE,
                   lightning=[0.35, 2.3]),
        event_shot(28.4, 3.2, "heat", "AUGUST 5  ·  JAPAN", "41.8°C",
                   "JAPAN'S HOTTEST TEMPERATURE EVER", size=260),
        event_shot(31.6, 3.2, "fire", "AUGUST  ·  EUROPE", "1,000,000",
                   "HECTARES BURNED ACROSS THE EU"),
        Shot(34.8, 1.8, "black", [
            Cue("AND THEN", 0.1, 0.9, size=110, tracking=24),
            Cue("CAME MELISSA", 0.95, 1.75, size=110, tracking=24),
        ], fin=0.0, fout=0.05),
        Shot(36.6, 4.4, "hurricane", [
            slam("185 MPH", 0.12, size=270, glow=(ICE, 26, 1), big=True),
            Cue("OCTOBER 28  ·  JAMAICA", 0.6, y=330, **TAG),
            Cue("CATEGORY 5. THE ISLAND'S FIRST.", 1.3, y=720, **SUB),
        ], push=0.04),
        event_shot(41.0, 3.4, "rain", "NOVEMBER  ·  SRI LANKA", "CYCLONE DITWAH",
                   "THE WORST FLOODS IN DECADES", glow_col=ICE, size=170,
                   lightning=[1.6]),
        Shot(44.4, 4.4, "black", [
            Cue("BEHIND EVERY NUMBER", 0.5, 2.1, face=CINZEL, size=52, style="track",
                tracking=10),
            Cue("IS A PLACE SOMEONE CALLS HOME", 2.3, 4.1, face=CINZEL, size=52,
                style="track", tracking=8),
        ], fin=0.3, fout=0.3),
    ]
    # accelerating city montage
    t, durs = 48.8, np.geomspace(0.8, 0.34, len(WARM_CITIES))
    for c, d in zip(WARM_CITIES, durs):
        a = CITY_ANOM[c]
        shots.append(Shot(t, float(d), "city", [
            slam(c.upper(), 0.0, size=170, y=500, glow=(HOT, 16, 1), soft=True),
            Cue(f"+{a:.1f}°C ABOVE NORMAL", 0.05, size=62, tracking=10, y=660,
                color=(1.0, 0.45, 0.32)),
        ], fin=0.0, fout=0.02, push=0.08, city=c))
        t += float(d)
    shots += [
        Shot(t, 0.8, "black", [], fin=0, fout=0),
    ]
    t += 0.8
    trio = t
    shots += [
        Shot(t, 3.6, "black", [
            slam("2023.", 0.0, size=150, x=500, y=500, glow=(HOT, 16, 1)),
            slam("2024.", 0.8, size=150, x=960, y=500, glow=(HOT, 16, 1)),
            slam("2025.", 1.6, size=150, x=1420, y=500, glow=(EMBER, 24, 2), big=True),
            Cue("THE THREE HOTTEST YEARS EVER RECORDED", 2.3, size=58, tracking=12,
                y=680, color=(0.95, 0.8, 0.6)),
        ], fin=0, fout=0.25),
    ]
    t += 3.6
    title = t
    shots += [
        Shot(t, 7.0, "title", [
            slam("2025", 0.0, size=360, y=500, glow=(EMBER, 40, 2), big=True),
            Cue("THE YEAR THE HEAT STAYED", 1.3, size=70, tracking=16, y=730,
                style="track", color=(1.0, 0.86, 0.66)),
        ], fin=0.0, fout=0.6, push=0.05),
    ]
    t += 7.0
    credits = t
    billing = dict(face=BEBAS, size=30, tracking=5, color=(0.6, 0.6, 0.64))
    shots += [
        Shot(t, 7.0, "black", [
            Cue("NOW PLAYING", 0.3, size=120, tracking=26, y=380),
            Cue("EVERYWHERE", 0.8, size=120, tracking=26, y=500, color=(1.0, 0.5, 0.3),
                glow=(HOT, 14, 1)),
            Cue("PLANET EARTH PRESENTS   A FILM BY EIGHT BILLION PEOPLE", 1.6, y=660,
                **billing),
            Cue("STARRING  HURRICANE MELISSA  ·  CYCLONE DITWAH  ·  THE SANTA ANA WINDS"
                "  ·  THE MONSOON  ·  AND LA NIÑA", 1.8, y=710, **billing),
            Cue("DATA  NASA GISTEMP V4  ·  METEOSTAT  ·  COPERNICUS C3S  ·  NOAA NCEI & NHC"
                "  ·  EU JRC / EFFIS  ·  WMO  ·  JMA", 2.0, y=760, **billing),
        ], fin=0.4, fout=1.0, push=0.0),
    ]
    t += 7.0
    return shots, dict(trio=trio, title=title, credits=credits, end=t)


SHOTS, MARKS = build_shots()
TOTAL = MARKS["end"]
HITS = sorted({(round(s.start + c.t_in, 3), c.big) for s in SHOTS for c in s.cues
               if c.hit and not c.soft})
SOFT_HITS = sorted(round(s.start + c.t_in, 3) for s in SHOTS for c in s.cues if c.soft)


def shot_at(T):
    for s in SHOTS:
        if s.start <= T < s.start + s.dur:
            return s
    return SHOTS[-1]


# --------------------------------------------------------------------------
# frame rendering
# --------------------------------------------------------------------------


def render_frame(T, fidx=0):
    shot = shot_at(T)
    t = T - shot.start
    bg = BACKDROPS[shot.bg](t, shot.params)
    zoom = 1 + shot.push * t / shot.dur
    img = Image.fromarray((clamp(bg) * 255).astype(np.uint8))
    cw, ch = BW / zoom, BH / zoom
    box = ((BW - cw) / 2, (BH - ch) / 2, (BW + cw) / 2, (BH + ch) / 2)
    frame = np.asarray(img.resize((W, H), Image.BILINEAR, box=box), np.float32) / 255
    frame = frame.copy()

    if shot.bg in ("fire", "title"):
        draw_embers(frame, T)

    for c in shot.cues:
        c.draw(frame, t, shot.dur)

    env = min(ease(t / shot.fin) if shot.fin > 0 else 1.0,
              ease((shot.dur - t) / shot.fout) if shot.fout > 0 else 1.0)
    frame *= env

    flash, shake = 0.0, 0.0
    for h, big in HITS:
        dt = T - h
        if 0 <= dt < 1.0:
            flash += (0.75 if big else 0.45) * math.exp(-dt / 0.07)
            shake += (22 if big else 12) * math.exp(-dt / 0.18)
    for h in SOFT_HITS:
        dt = T - h
        if 0 <= dt < 0.5:
            flash += 0.2 * math.exp(-dt / 0.05)
            shake += 6 * math.exp(-dt / 0.1)
    if flash > 0.01:
        frame += min(flash, 1.0) * (1 - frame) * np.array([1.0, 0.93, 0.85], np.float32)
    if shake > 0.5:
        dx = int(shake * math.sin(T * 91.0))
        dy = int(shake * math.cos(T * 67.0))
        frame = np.roll(frame, (dy, dx), axis=(0, 1))

    frame *= VIGNETTE
    frame += GRAIN[fidx % len(GRAIN)]
    frame[:BAR] = 0
    frame[H - BAR:] = 0
    return (clamp(frame) * 255).astype(np.uint8)


def draw_embers(frame, T):
    e = EMBERS
    ys = (e["y"] - e["v"] * T) % 1.0
    xs = (e["x"] + 0.015 * np.sin(T * 1.7 + e["ph"])) % 1.0
    for x, y, s, ph in zip(xs, ys, e["s"], e["ph"]):
        px, py = int(x * W), int(BAR + y * (H - 2 * BAR))
        r = int(2 + 3 * s)
        if r <= px < W - r and r <= py < H - r:
            tw = 0.55 + 0.45 * math.sin(T * 9 + ph * 3)
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            blob = np.exp(-(xx ** 2 + yy ** 2) / (0.35 * r * r))[..., None] * tw
            reg = frame[py - r:py + r + 1, px - r:px + r + 1]
            reg += blob * np.array([1.0, 0.55, 0.15], np.float32)


def render_chunk(args):
    idx, frames = args
    path = os.path.join(BUILD, f"trailer_{idx:02d}.mp4")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(render_frame(f / FPS, f).tobytes())
    proc.stdin.close()
    proc.wait()
    return path


# --------------------------------------------------------------------------
# sound design
# --------------------------------------------------------------------------


def lp(x, hz, order=2):
    b, a = signal.butter(order, hz / (SR / 2), "low")
    return signal.lfilter(b, a, x)


def hp(x, hz, order=2):
    b, a = signal.butter(order, hz / (SR / 2), "high")
    return signal.lfilter(b, a, x)


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band")
    return signal.lfilter(b, a, x)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def saw(f, t, detune=0.0):
    return signal.sawtooth(2 * np.pi * f * (1 + detune) * t)


def braam(big=False):
    t = tvec(4.5 if big else 3.2)
    notes = [55.0, 82.41, 110.0, 130.81] + ([41.2] if big else [])
    x = sum(saw(f, t, d) for f in notes for d in (-0.004, 0.0, 0.005))
    x = np.tanh(x * 0.6)
    bright = lp(x, 2400) * np.exp(-t / 0.35)
    dark = lp(x, 380) * np.exp(-t / (1.6 if big else 1.1))
    env = np.minimum(1, t / 0.015)
    return (bright * 0.5 + dark) * env * (1.0 if big else 0.8)


def boom(big=False):
    t = tvec(3.0)
    f = 32 + 90 * np.exp(-t / 0.18)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.9 if big else 0.6))
    noise = RNG.normal(0, 1, len(t))
    thud = lp(noise, 180) * np.exp(-t / 0.09) * 3
    click = hp(noise, 2500) * np.exp(-t / 0.012) * 0.6
    return sub * 1.2 + thud + click


def riser(dur):
    t = tvec(dur)
    k = t / dur
    noise = hp(RNG.normal(0, 1, len(t)), 1200) * 0.25
    f = 180 * 2 ** (k * 2.5)
    tone = sum(np.sin(2 * np.pi * np.cumsum(f * m) / SR) / m for m in (1, 2, 4))
    return (noise + tone * 0.5) * k ** 2.2


def whoosh(dur=0.8):
    t = tvec(dur)
    env = np.sin(np.pi * t / dur) ** 2
    return bp(RNG.normal(0, 1, len(t)), 300, 3000) * env * 0.5


def tick():
    t = tvec(0.06)
    return (hp(RNG.normal(0, 1, len(t)), 3000) * np.exp(-t / 0.004)
            + np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.01) * 0.6)


def heartbeat():
    t = tvec(0.6)
    thump = lambda d: np.sin(2 * np.pi * 48 * (t - d)) * np.exp(-np.maximum(0, t - d) / 0.09) * (t >= d)  # noqa: E731
    return thump(0.0) + 0.7 * thump(0.24)


def piano(freq, dur=3.5):
    t = tvec(dur)
    x = sum(np.sin(2 * np.pi * freq * h * (1 + 0.0004 * h * h) * t) * np.exp(-t * (0.7 + 0.9 * h)) / h
            for h in range(1, 7))
    return x * np.minimum(1, t / 0.004) * 0.5


def thunder(dur=2.5):
    t = tvec(dur)
    env = np.minimum(1, t / 0.05) * np.exp(-t / 0.8)
    return lp(RNG.normal(0, 1, len(t)), 220, 3) * env * 5


def make_audio(path):
    n = int(TOTAL * SR) + SR
    mix = np.zeros(n)

    def add(x, at, gain=1.0):
        i = int(at * SR)
        j = min(n, i + len(x))
        if j > i:
            mix[i:j] += x[:j - i] * gain

    t = np.arange(n) / SR
    # drone bed with automation
    pts = [(0, 0), (3, 0.3), (8.6, 0.45), (13, 0.75), (34.8, 0.9), (41, 0.8), (44.4, 0.1),
           (48.8, 0.55), (MARKS["trio"] - 0.8, 0.8), (MARKS["trio"] - 0.75, 0),
           (MARKS["trio"], 0.5), (MARKS["title"], 1.0), (MARKS["credits"], 0.7),
           (TOTAL, 0)]
    level = np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
    drone = lp(sum(saw(f, t, d) for f in (55.0, 82.41) for d in (-0.003, 0.003)), 260)
    drone *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    pad = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 659.25, 523.25)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    mix += (drone * 0.25 + pad * 0.02) * level

    add(boom(), 0.3, 0.35)
    for k in range(4):
        add(heartbeat(), 4.6 + k * 1.0, 0.8)
    for k in range(8):
        add(tick(), 9.0 + k * 0.5, 0.25)
    add(riser(2.2), 13.0 - 2.2, 0.5)
    add(riser(3.5), 36.6 + 0.12 - 3.5, 0.55)
    add(riser(5.0), MARKS["trio"] - 0.8 - 5.0, 0.5)
    for h, big in HITS:
        add(braam(big), h, 0.55 if big else 0.4)
        add(boom(big), h, 0.9 if big else 0.6)
    # extra weight on the title card
    add(braam(True), MARKS["title"], 0.4)
    # whooshes on tag lines and ambience
    for s in SHOTS:
        for c in s.cues:
            if not c.hit and s.bg != "black" and c.face == BEBAS:
                add(whoosh(0.7), s.start + c.t_in - 0.35, 0.35)
        if s.bg == "rain":
            amb = bp(RNG.normal(0, 1, int(s.dur * SR)), 800, 6000) * 0.12
            add(amb * np.minimum(1, tvec(s.dur) / 0.1), s.start)
            for lt in s.params.get("lightning", []):
                add(thunder(), s.start + lt, 0.5)
        if s.bg in ("fire",):
            cr = np.zeros(int(s.dur * SR))
            for _ in range(int(s.dur * 25)):
                i = RNG.integers(0, len(cr) - 400)
                cr[i:i + 400] += hp(RNG.normal(0, 1, 400), 1500) * np.exp(-np.arange(400) / 60)
            add(cr * 0.15 + lp(RNG.normal(0, 1, len(cr)), 400) * 0.1, s.start)
        if s.bg == "hurricane":
            wn = bp(RNG.normal(0, 1, int(s.dur * SR)), 200, 1200)
            add(wn * (0.25 + 0.15 * np.sin(2 * np.pi * 0.7 * tvec(s.dur))), s.start)
        if s.bg == "city":
            add(tick(), s.start, 0.6)
            add(boom(), s.start, 0.3)
    # the quiet beat
    for f, at in [(220.0, 44.8), (329.63, 45.5), (523.25, 46.2), (174.61, 47.0),
                  (261.63, 47.0), (440.0, 47.0)]:
        add(piano(f), at, 0.35)

    # reverb: convolve with a decaying-noise impulse response
    ir_t = tvec(2.2)
    wet = []
    for seed in (1, 2):
        ir = np.random.default_rng(seed).normal(0, 1, len(ir_t)) * np.exp(-ir_t / 0.55)
        ir = lp(ir, 5000)
        wet.append(signal.fftconvolve(mix, ir)[:n] * 0.012)
    left, right = mix + wet[0], mix + wet[1]
    stereo = np.stack([left, right], axis=1)
    fade = np.clip((TOTAL - t) / 1.0, 0, 1)[:, None]
    stereo *= fade
    stereo = np.tanh(stereo / np.percentile(np.abs(stereo), 99.9) * 1.5) * 0.97
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(BUILD, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "--still":
        for s in sys.argv[2:]:
            T = float(s)
            Image.fromarray(render_frame(T)).save(os.path.join(BUILD, f"trailer_{T:05.1f}.png"))
        return
    nframes = int(round(TOTAL * FPS))
    workers = os.cpu_count() or 2
    size = math.ceil(nframes / workers)
    chunks = [(i, list(range(i * size, min(nframes, (i + 1) * size)))) for i in range(workers)]
    print(f"rendering {nframes} frames ({TOTAL:.1f}s) on {workers} workers")
    with Pool(workers) as pool:
        parts = pool.map(render_chunk, chunks)
    listing = os.path.join(BUILD, "trailer_parts.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    audio = os.path.join(BUILD, "trailer_audio.wav")
    make_audio(audio)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a",
                    "192k", "-shortest", "-movflags", "+faststart", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
