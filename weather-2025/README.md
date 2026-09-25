# 2025: The Year in Weather

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
