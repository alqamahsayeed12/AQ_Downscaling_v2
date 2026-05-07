from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from io import BytesIO
import base64
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from PIL import Image

from aq_downscaling_app.pipeline import PipelineConfig, run_pipeline


CAIRO_DEFAULTS = {
    "lat_min": 29.0444,
    "lat_max": 31.0444,
    "lon_min": 30.2357,
    "lon_max": 32.2357,
}


def init_state() -> None:
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_netcdf", None)
    st.session_state.setdefault("last_result", None)


def log_message(message: str) -> None:
    st.session_state.logs.append(message)
    st.session_state.log_box.code("\n".join(st.session_state.logs[-200:]), language="text")


def render_map(netcdf_path: str) -> None:
    with st.spinner("Preparing interactive map frames..."):
        payload = build_map_payload(netcdf_path)
    components.html(build_leaflet_viewer(payload), height=720)


@st.cache_data(show_spinner=False)
def build_map_payload(netcdf_path: str) -> dict:
    ds = xr.open_dataset(netcdf_path)
    try:
        times = pd.to_datetime(ds.time.values)
        pm25 = ds["DS_BC_DNN_PM25"]
        lon_min = float(pm25.lon.min())
        lon_max = float(pm25.lon.max())
        lat_min = float(pm25.lat.min())
        lat_max = float(pm25.lat.max())
        frames = []
        for time_index, forecast_time in enumerate(times):
            data = pm25.isel(time=time_index).values
            frames.append(
                {
                    "time": forecast_time.strftime("%Y-%m-%d %H:%M"),
                    "image": array_to_png_data_url(data),
                    "min": f"{float(np.nanmin(data)):.2f}",
                    "mean": f"{float(np.nanmean(data)):.2f}",
                    "max": f"{float(np.nanmax(data)):.2f}",
                }
            )

        return {
            "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
            "center": [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
            "frames": frames,
        }
    finally:
        ds.close()


def build_leaflet_viewer(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: transparent;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111827;
    }}
    .viewer {{
      position: relative;
      height: 700px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      overflow: hidden;
      background: #e5e7eb;
    }}
    #map {{
      height: 100%;
      width: 100%;
    }}
    .toolbar {{
      position: absolute;
      top: 12px;
      left: 12px;
      right: 12px;
      z-index: 1000;
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 8px;
      align-items: center;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(17, 24, 39, 0.12);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.16);
      padding: 8px;
      backdrop-filter: blur(8px);
    }}
    .buttons {{
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    button, select {{
      height: 32px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      background: #ffffff;
      color: #111827;
      font-size: 13px;
      line-height: 1;
    }}
    button {{
      min-width: 34px;
      padding: 0 10px;
      cursor: pointer;
    }}
    button:hover {{
      background: #f3f4f6;
    }}
    .timeline {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      min-width: 0;
    }}
    #timeSlider {{
      width: 100%;
      accent-color: #2563eb;
    }}
    #timestamp {{
      min-width: 132px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .viewer-controls {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .opacity {{
      display: flex;
      gap: 6px;
      align-items: center;
      font-size: 12px;
      white-space: nowrap;
    }}
    #opacitySlider {{
      width: 92px;
      accent-color: #2563eb;
    }}
    .stats {{
      position: absolute;
      left: 12px;
      bottom: 14px;
      z-index: 1000;
      display: flex;
      gap: 6px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(17, 24, 39, 0.12);
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.14);
      padding: 7px;
    }}
    .stat {{
      min-width: 72px;
      padding: 4px 6px;
      border-right: 1px solid #e5e7eb;
    }}
    .stat:last-child {{
      border-right: 0;
    }}
    .stat-label {{
      color: #6b7280;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .stat-value {{
      margin-top: 2px;
      font-size: 14px;
      font-weight: 800;
    }}
    .legend {{
      position: absolute;
      right: 18px;
      bottom: 28px;
      z-index: 1000;
      width: 138px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(15, 23, 42, 0.18);
      border-radius: 6px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.24);
      padding: 7px 8px 6px;
      line-height: 1.1;
    }}
    .legend-title {{
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 5px;
    }}
    .legend-ramp {{
      height: 9px;
      width: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #1d4ed8, #06b6d4, #34d399, #bef264, #facc15, #f97316, #dc2626);
    }}
    .legend-ticks {{
      display: flex;
      justify-content: space-between;
      margin-top: 4px;
      font-size: 9px;
    }}
    @media (max-width: 760px) {{
      .viewer {{
        height: 640px;
      }}
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .viewer-controls {{
        flex-wrap: wrap;
      }}
      .stats {{
        right: 12px;
        bottom: 76px;
      }}
    }}
  </style>
</head>
<body>
  <div class="viewer">
    <div id="map"></div>
    <div class="toolbar">
      <div class="buttons">
        <button id="prevBtn" title="Previous timestep">Prev</button>
        <button id="playBtn" title="Play or pause">Play</button>
        <button id="nextBtn" title="Next timestep">Next</button>
        <button id="resetBtn" title="Reset to first timestep">Reset</button>
      </div>
      <div class="timeline">
        <input id="timeSlider" type="range" min="0" value="0" step="1">
        <div id="timestamp"></div>
      </div>
      <div class="viewer-controls">
        <select id="basemapSelect" title="Basemap">
          <option value="positron" selected>Light</option>
          <option value="osm">OpenStreetMap</option>
          <option value="dark">Dark</option>
          <option value="imagery">Imagery</option>
        </select>
        <label class="opacity">Opacity
          <input id="opacitySlider" type="range" min="0" max="1" value="0.72" step="0.05">
        </label>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><div class="stat-label">Min</div><div id="statMin" class="stat-value"></div></div>
      <div class="stat"><div class="stat-label">Mean</div><div id="statMean" class="stat-value"></div></div>
      <div class="stat"><div class="stat-label">Max</div><div id="statMax" class="stat-value"></div></div>
    </div>
    <div class="legend">
      <div class="legend-title">PM2.5</div>
      <div class="legend-ramp"></div>
      <div class="legend-ticks"><span>0</span><span>60</span><span>120</span></div>
    </div>
  </div>
  <script>
    const payload = {payload_json};
    const frames = payload.frames;
    const bounds = payload.bounds;
    const map = L.map("map", {{ zoomControl: true, preferCanvas: true }}).fitBounds(bounds);
    L.control.scale({{ position: "bottomleft" }}).addTo(map);

    const basemaps = {{
      positron: L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO"
      }}),
      osm: L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        attribution: "&copy; OpenStreetMap contributors"
      }}),
      dark: L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO"
      }}),
      imagery: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
        attribution: "Tiles &copy; Esri"
      }})
    }};
    let activeBasemap = basemaps.positron.addTo(map);
    const overlay = L.imageOverlay(frames[0].image, bounds, {{ opacity: 0.72, zIndex: 10 }}).addTo(map);
    L.rectangle(bounds, {{ color: "#111827", weight: 1, fill: false }}).addTo(map);

    frames.forEach((frame) => {{
      const img = new Image();
      img.src = frame.image;
    }});

    let currentIndex = 0;
    let timer = null;
    const slider = document.getElementById("timeSlider");
    const timestamp = document.getElementById("timestamp");
    const playBtn = document.getElementById("playBtn");
    const opacitySlider = document.getElementById("opacitySlider");
    slider.max = String(frames.length - 1);

    function renderFrame(index) {{
      currentIndex = Math.max(0, Math.min(frames.length - 1, index));
      const frame = frames[currentIndex];
      overlay.setUrl(frame.image);
      slider.value = String(currentIndex);
      timestamp.textContent = frame.time;
      document.getElementById("statMin").textContent = frame.min;
      document.getElementById("statMean").textContent = frame.mean;
      document.getElementById("statMax").textContent = frame.max;
    }}

    function stopPlayback() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
      playBtn.textContent = "Play";
    }}

    function startPlayback() {{
      stopPlayback();
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        if (currentIndex >= frames.length - 1) {{
          stopPlayback();
          return;
        }}
        renderFrame(currentIndex + 1);
      }}, 850);
    }}

    document.getElementById("prevBtn").addEventListener("click", () => {{
      stopPlayback();
      renderFrame(currentIndex - 1);
    }});
    document.getElementById("nextBtn").addEventListener("click", () => {{
      stopPlayback();
      renderFrame(currentIndex + 1);
    }});
    document.getElementById("resetBtn").addEventListener("click", () => {{
      stopPlayback();
      renderFrame(0);
    }});
    playBtn.addEventListener("click", () => {{
      if (timer === null) {{
        if (currentIndex >= frames.length - 1) renderFrame(0);
        startPlayback();
      }} else {{
        stopPlayback();
      }}
    }});
    slider.addEventListener("input", (event) => {{
      stopPlayback();
      renderFrame(Number(event.target.value));
    }});
    opacitySlider.addEventListener("input", (event) => {{
      overlay.setOpacity(Number(event.target.value));
    }});
    document.getElementById("basemapSelect").addEventListener("change", (event) => {{
      map.removeLayer(activeBasemap);
      activeBasemap = basemaps[event.target.value].addTo(map);
      activeBasemap.bringToBack();
    }});

    renderFrame(0);
  </script>
</body>
</html>
"""


def array_to_png_data_url(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype="float32")
    arr = np.flipud(arr)
    vmin, vmax = 0.0, 120.0
    scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap("turbo")
    rgba = (cmap(scaled) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(arr), 255, 0).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    st.set_page_config(page_title="AQ Downscaling", layout="wide")
    init_state()

    st.title("AQ Downscaling")
    st.caption("GEOS-FP PM2.5 bias correction and 5 km downscaling through NCCS OPeNDAP")

    with st.sidebar:
        st.header("Run Settings")
        default_date = date.today() - timedelta(days=1)
        run_date = st.date_input("Forecast initialization date", value=default_date)
        date_str = run_date.strftime("%Y%m%d")

        st.subheader("Domain")
        preset = st.selectbox("Preset", ["Cairo 2° x 2°", "Custom"], index=0)
        if preset == "Cairo 2° x 2°":
            lat_min = st.number_input("Latitude min", value=CAIRO_DEFAULTS["lat_min"], format="%.4f")
            lat_max = st.number_input("Latitude max", value=CAIRO_DEFAULTS["lat_max"], format="%.4f")
            lon_min = st.number_input("Longitude min", value=CAIRO_DEFAULTS["lon_min"], format="%.4f")
            lon_max = st.number_input("Longitude max", value=CAIRO_DEFAULTS["lon_max"], format="%.4f")
        else:
            lat_min = st.number_input("Latitude min", value=CAIRO_DEFAULTS["lat_min"], format="%.4f")
            lat_max = st.number_input("Latitude max", value=CAIRO_DEFAULTS["lat_max"], format="%.4f")
            lon_min = st.number_input("Longitude min", value=CAIRO_DEFAULTS["lon_min"], format="%.4f")
            lon_max = st.number_input("Longitude max", value=CAIRO_DEFAULTS["lon_max"], format="%.4f")

        st.subheader("Outputs")
        output_dir = st.text_input("Output folder", value="./OUT_DS")
        write_geotiff = st.checkbox("Write GeoTIFFs", value=True)
        keep_intermediate = st.checkbox("Keep 25 km bias-corrected NetCDF", value=True)
        batch_size = st.number_input("Prediction batch size", min_value=1, value=8192, step=1024)

        run = st.button("Run Downscaling", type="primary")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Progress")
        st.session_state.log_box = st.empty()
        if st.session_state.logs:
            st.session_state.log_box.code("\n".join(st.session_state.logs[-200:]), language="text")
        else:
            st.session_state.log_box.info("Run logs will appear here.")

        if run:
            st.session_state.logs = []
            st.session_state.log_box.code("", language="text")
            config = PipelineConfig(
                run_date=date_str,
                lat_min=float(lat_min),
                lat_max=float(lat_max),
                lon_min=float(lon_min),
                lon_max=float(lon_max),
                output_dir=Path(output_dir),
                write_geotiff=write_geotiff,
                keep_intermediate=keep_intermediate,
                batch_size=int(batch_size),
            )
            try:
                with st.spinner("Running downscaling workflow..."):
                    result = run_pipeline(config, logger=log_message)
                st.session_state.last_result = result
                st.session_state.last_netcdf = str(result.final_netcdf)
                st.success("Run complete.")
            except Exception as exc:
                st.error(f"Run failed: {exc}")
                log_message(f"ERROR: {exc}")

        result = st.session_state.last_result
        if result:
            st.subheader("Generated Files")
            st.write(f"NetCDF: `{result.final_netcdf}`")
            if result.bias_corrected_netcdf:
                st.write(f"Bias-corrected NetCDF: `{result.bias_corrected_netcdf}`")
            if result.geotiff_dir:
                st.write(f"GeoTIFF directory: `{result.geotiff_dir}` ({result.geotiff_count} files)")
            st.write(f"Elapsed: `{result.elapsed_seconds:.1f}s`")

    with right:
        st.subheader("5 km Bias-Corrected PM2.5")
        if st.session_state.last_netcdf and Path(st.session_state.last_netcdf).exists():
            render_map(st.session_state.last_netcdf)
        else:
            st.info("Run the workflow or place a generated NetCDF in the selected output folder to view the map.")


if __name__ == "__main__":
    main()
