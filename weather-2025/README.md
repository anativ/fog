# 2025: The Year in Weather

## Trailer cut

`weather_2025_trailer.mp4` is a 73-second, movie-trailer take on the same year. It is letterboxed at 2.4:1 with film grain and animated backdrops for fire, rain, heat and a spinning hurricane. Big numbers slam in with a flash and a shake. The sound is trailer-style audio made in code: drones, brass hits, risers, ticks and a quiet piano beat. It ends on a title card and a billing block.

The cut runs: *based on real weather data* → 146 years of warming stripes → *the hottest years ever recorded* → LA fires (16,000 structures) → Pakistan floods (3M+ people in need) → 46°C in Iberia → Texas Hill Country flood (20 in of rain) → Japan 41.8°C → 1,000,000 ha burned in the EU → Hurricane Melissa (185 mph) → Cyclone Ditwah → *behind every number…* → accelerating city cuts (Moscow +2.1°C …) → 2023. 2024. 2025. → **2025: The Year the Heat Stayed**.

```sh
pip install numpy pandas pillow scipy matplotlib imageio-ffmpeg
python3 prepare_data.py
python3 make_trailer.py                 # renders weather_2025_trailer.mp4 (~2.5 min on 4 cores)
python3 make_trailer.py --still 38.5    # preview frames at given seconds
```

Fonts in `fonts/` (Anton, Bebas Neue, Cinzel) are from Google Fonts under the SIL Open Font License.

## Summary cut

`weather_2025_summary.mp4` is an 83-second 1080p video summarizing the weather of 2025.

1. **Global temperature, 1880–2025.** NASA GISTEMP v4 annual anomalies. 2025 was 2nd-warmest in NASA's data and 3rd-warmest per NOAA and Copernicus, about 1.47°C above pre-industrial levels.
2. **2025 month by month**, compared with 2023 and 2024.
3. **12 cities.** Daily 2025 temperatures (7-day mean) against each station's 1991–2020 normal, from Meteostat station data.
4. **Nine defining events.** LA wildfires, Pakistan floods, European heat, the Texas Hill Country flood, Japan's 41.8°C record, the EU wildfire season, Hurricane Melissa, Cyclone Ditwah, and the Atlantic season with no U.S. hurricane landfall.
5. **Close.** 2023–2025 was the first three-year period averaging above 1.5°C (Copernicus).

## Rebuild

```sh
pip install numpy pandas matplotlib imageio-ffmpeg
python3 prepare_data.py   # downloads GISTEMP + Meteostat, writes data/
python3 make_video.py     # renders weather_2025_summary.mp4
python3 make_video.py --still <scene> <seconds>   # preview a single frame
```

The soundtrack is an ambient pad synthesized in `make_video.py`.
