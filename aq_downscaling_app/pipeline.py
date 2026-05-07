from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import keras
import keras as k
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENDAP_BASE = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast"
MODEL_MULTIPLE = 40
MODEL_MIN_LAT_SPAN = 10.5
MODEL_MIN_LON_SPAN = 13.0

FEATURE_COLUMNS = [
    "WIND",
    "PS",
    "Q500",
    "QV10M",
    "T10M",
    "T500",
    "U10M",
    "V10M",
    "BCSMASS",
    "DUSMASS25",
    "OCSMASS",
    "SO2SMASS",
    "SO4SMASS",
    "SSSMASS25",
    "TOTEXTTAU",
]
DNN_COLUMNS = [f"DNN_{i:02d}" for i in range(10)]
FEATURE_COLUMNS_ENSEMBLE = FEATURE_COLUMNS + DNN_COLUMNS
OUT_COLUMNS = ["lat", "lon", "time", "GEOSPM25", "BC_DNN_PM25"]

NETCDF_ENCODING_BC = {
    "lat": {"_FillValue": None, "zlib": True, "dtype": "float32"},
    "lon": {"_FillValue": None, "zlib": True, "dtype": "float32"},
    "GEOSPM25": {"_FillValue": -999, "zlib": True, "dtype": "float32"},
    "BC_DNN_PM25": {"_FillValue": -999, "zlib": True, "dtype": "float32"},
}
NETCDF_ENCODING_DS = {
    "lat": {"_FillValue": None, "zlib": True, "dtype": "float32"},
    "lon": {"_FillValue": None, "zlib": True, "dtype": "float32"},
    "DS_GEOSPM25": {"_FillValue": -999, "zlib": True, "dtype": "float32"},
    "DS_BC_DNN_PM25": {"_FillValue": -999, "zlib": True, "dtype": "float32"},
}


@dataclass(frozen=True)
class PipelineConfig:
    run_date: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    output_dir: Path
    write_geotiff: bool = True
    keep_intermediate: bool = True
    batch_size: int = 8192


@dataclass(frozen=True)
class PipelineResult:
    final_netcdf: Path
    bias_corrected_netcdf: Path | None
    cached_input: Path
    geotiff_dir: Path | None
    geotiff_count: int
    elapsed_seconds: float


class Timer:
    def __init__(self, logger: Callable[[str], None]) -> None:
        self.logger = logger
        self.started = time.perf_counter()
        self.last = self.started

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self.logger(f"[timing] {label}: +{now - self.last:.1f}s, total {now - self.started:.1f}s")
        self.last = now


def custom_loss(o, p):
    ioa = 1 - (k.backend.sum((o - p) ** 2)) / (
        k.backend.sum((k.backend.abs(p - k.backend.mean(o)) + k.backend.abs(o - k.backend.mean(o))) ** 2)
    )
    return -ioa


keras.losses.customLoss1 = custom_loss


def validate_date(date_str: str) -> None:
    try:
        run_date = datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError("Date must use YYYYMMDD format.") from exc
    if run_date > date.today():
        raise ValueError("Date cannot be in the future.")


def validate_domain(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> None:
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= min < max <= 90.")
    if not (-180 <= lon_min < lon_max <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= min < max <= 180.")


def expand_bounds_for_model(
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    lat_center = sum(lat_bounds) / 2
    lon_center = sum(lon_bounds) / 2
    lat_span = max(lat_bounds[1] - lat_bounds[0], MODEL_MIN_LAT_SPAN)
    lon_span = max(lon_bounds[1] - lon_bounds[0], MODEL_MIN_LON_SPAN)
    return (
        (max(-90.0, lat_center - lat_span / 2), min(90.0, lat_center + lat_span / 2)),
        (max(-180.0, lon_center - lon_span / 2), min(180.0, lon_center + lon_span / 2)),
    )


def trim_to_multiple(ds: xr.Dataset, dim: str, multiple: int) -> xr.Dataset:
    size = ds.sizes[dim]
    remainder = size % multiple
    if remainder == 0:
        return ds
    trim_start = remainder // 2
    trim_end = remainder - trim_start
    return ds.isel({dim: slice(trim_start, size - trim_end)})


def normalize(df: pd.DataFrame, mx: pd.Series, mn: pd.Series) -> pd.DataFrame:
    values = (df.values - mn.values) / (mx.values - mn.values)
    values[values < 0] = np.nan
    return pd.DataFrame(values, columns=df.columns, index=df.index)


def load_scalars() -> tuple[pd.Series, pd.Series]:
    mx_mn = pd.read_csv(PROJECT_ROOT / "Scalars" / "max_min4.csv", index_col=0).T
    mx_mn.loc["mx", DNN_COLUMNS] = 1000.0
    mx_mn.loc["mn", DNN_COLUMNS] = 0.0
    return mx_mn.loc["mx", :], mx_mn.loc["mn", :]


def load_models(logger: Callable[[str], None]) -> tuple[dict[int, list], dict[int, object], object]:
    model_dir = PROJECT_ROOT / "Model"
    custom_objects = {"TFOpLambda": lambda x: x, "customLoss1": custom_loss}
    bias_models: dict[int, list] = {}
    ensemble_models: dict[int, object] = {}

    logger("Loading bias-correction models...")
    with k.utils.custom_object_scope(custom_objects):
        for day_num in (1, 2, 3):
            bias_models[day_num] = [
                k.models.load_model(model_dir / f"v1_4_dnn_bias_Correction_day{day_num}_fold{fold:02d}.h5")
                for fold in range(10)
            ]
            ensemble_models[day_num] = k.models.load_model(
                model_dir / f"v1_4_day{day_num}_dnn_bias_Correction_ensemble.h5"
            )

    logger("Loading downscaling model...")
    downscale_model = keras.models.load_model(model_dir / "model_downscale_v1.h5", custom_objects={"customLoss1": custom_loss})
    return bias_models, ensemble_models, downscale_model


def domain_cache_name(date_str: str, lat_bounds: tuple[float, float], lon_bounds: tuple[float, float]) -> str:
    def fmt(value: float) -> str:
        return f"{value:.4f}".replace("-", "m").replace(".", "p")

    return f"opendap_{date_str}_lat{fmt(lat_bounds[0])}_{fmt(lat_bounds[1])}_lon{fmt(lon_bounds[0])}_{fmt(lon_bounds[1])}.nc"


def opendap_url(product: str, date_str: str) -> str:
    return f"{OPENDAP_BASE}/{product}/{product}.{date_str}_00"


def coordinate_slice(values, lower: float, upper: float) -> slice:
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def normalize_lon_bounds(ds: xr.Dataset, lon_bounds: tuple[float, float]) -> tuple[float, float]:
    lon_min = float(ds.lon.min())
    lon_max = float(ds.lon.max())
    lower, upper = lon_bounds
    if lon_min >= 0 and lower < 0:
        lower = lower % 360
        upper = upper % 360
    if lon_max <= 180 and lower > 180:
        lower = ((lower + 180) % 360) - 180
        upper = ((upper + 180) % 360) - 180
    return lower, upper


def subset_opendap_dataset(
    product: str,
    date_str: str,
    variables: list[str],
    time_indices,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    logger: Callable[[str], None],
) -> xr.Dataset:
    url = opendap_url(product, date_str)
    logger(f"Opening OPeNDAP product: {url}")
    with xr.open_dataset(url) as remote:
        missing = sorted(set(variables) - set(remote.data_vars))
        if missing:
            raise KeyError(f"Missing expected OPeNDAP variables in {product}: {missing}")
        lon_lower, lon_upper = normalize_lon_bounds(remote, lon_bounds)
        return (
            remote[variables]
            .isel(time=time_indices)
            .sel(
                lat=coordinate_slice(remote.lat.values, *lat_bounds),
                lon=coordinate_slice(remote.lon.values, lon_lower, lon_upper),
            )
            .load()
        )


def get_input_file_opendap(
    date_str: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    timer: Timer,
    logger: Callable[[str], None],
) -> Path:
    input_dir = PROJECT_ROOT / "IN_NetCDF"
    input_dir.mkdir(parents=True, exist_ok=True)
    cache_file = input_dir / domain_cache_name(date_str, lat_bounds, lon_bounds)
    if cache_file.exists():
        logger(f"Using cached OPeNDAP subset: {cache_file}")
        return cache_file

    met_vars = ["ps", "q500", "qv10m", "t10m", "t500", "u10m", "v10m"]
    aer_vars = ["bcsmass", "dusmass25", "ocsmass", "so2smass", "so4smass", "sssmass25", "totexttau", "nismass25"]
    met_time_indices = np.arange(1, 73, 3)
    aer_time_indices = np.arange(1, 25)

    met = subset_opendap_dataset("tavg1_2d_slv_Nx", date_str, met_vars, met_time_indices, lat_bounds, lon_bounds, logger)
    timer.mark("loaded meteorology OPeNDAP subset")
    aer = subset_opendap_dataset("tavg3_2d_aer_Nx", date_str, aer_vars, aer_time_indices, lat_bounds, lon_bounds, logger)
    timer.mark("loaded aerosol OPeNDAP subset")

    met = met.rename({name: name.upper() for name in met_vars})
    aer = aer.rename({name: name.upper() for name in aer_vars})
    merged = xr.merge([met, aer], compat="override").astype("float32")
    merged.attrs.update({"source": "NCCS GEOS-FP OPeNDAP subset", "date": date_str})
    merged.to_netcdf(cache_file)
    timer.mark("cached merged OPeNDAP subset")
    logger(f"Cached OPeNDAP subset: {cache_file}")
    return cache_file


def prepare_features(input_file: Path, lat_bounds: tuple[float, float], lon_bounds: tuple[float, float], logger: Callable[[str], None]) -> pd.DataFrame:
    logger("Opening GEOS NetCDF and preparing features...")
    with xr.open_dataset(input_file) as source:
        ds = source.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds)).load()

    before = (ds.sizes["lat"], ds.sizes["lon"])
    ds = trim_to_multiple(trim_to_multiple(ds, "lat", MODEL_MULTIPLE), "lon", MODEL_MULTIPLE)
    after = (ds.sizes["lat"], ds.sizes["lon"])
    if after[0] == 0 or after[1] == 0:
        raise ValueError("Selected model domain is too small after grid trimming.")
    if before != after:
        logger(f"Trimmed grid from lat/lon {before} to {after} to satisfy model multiples of 40.")

    df = ds.to_dataframe().reset_index()
    for column in ["NISMASS25", "OCSMASS", "SO2SMASS", "SO4SMASS", "SSSMASS25", "DUSMASS25", "BCSMASS"]:
        df[column] = df[column].astype("float32") * np.float32(1e9)

    df["GEOSPM25"] = (
        df["DUSMASS25"]
        + df["SSSMASS25"]
        + df["NISMASS25"]
        + df["BCSMASS"]
        + df["OCSMASS"] * np.float32(1.6)
        + df["SO4SMASS"] * np.float32(1.375)
    )
    df["WIND"] = np.sqrt(df["U10M"] ** 2 + df["V10M"] ** 2).astype("float32")
    df["Date"] = df["time"].dt.strftime("%Y%m%d")
    return df


def predict_bias_correction(
    df: pd.DataFrame,
    date_str: str,
    bias_models: dict[int, list],
    ensemble_models: dict[int, object],
    mx: pd.Series,
    mn: pd.Series,
    batch_size: int,
) -> xr.Dataset:
    base_date = pd.to_datetime(date_str, format="%Y%m%d")
    daily_frames = []

    for day_num in (1, 2, 3):
        day_str = (base_date + pd.Timedelta(days=day_num - 1)).strftime("%Y%m%d")
        day_df = df.loc[df["Date"] == day_str].copy()
        if day_df.empty:
            continue
        normalized = normalize(day_df[FEATURE_COLUMNS], mx[FEATURE_COLUMNS], mn[FEATURE_COLUMNS])
        for fold, model in enumerate(bias_models[day_num]):
            day_df[f"DNN_{fold:02d}"] = np.asarray(model.predict(normalized, batch_size=batch_size, verbose=0)).reshape(-1)
        ensemble_input = normalize(day_df[FEATURE_COLUMNS_ENSEMBLE], mx[FEATURE_COLUMNS_ENSEMBLE], mn[FEATURE_COLUMNS_ENSEMBLE])
        day_df["BC_DNN_PM25"] = np.asarray(ensemble_models[day_num].predict(ensemble_input, batch_size=batch_size, verbose=0)).reshape(-1)
        daily_frames.append(day_df)

    if not daily_frames:
        raise RuntimeError("No forecast-day records were available for bias correction.")
    output_df = pd.concat(daily_frames, axis=0)[OUT_COLUMNS].set_index(["lat", "lon", "time"])
    return xr.Dataset.from_dataframe(output_df).astype("float32")


def downscale(ds: xr.Dataset, model: object, batch_size: int) -> xr.Dataset:
    lat = np.arange(float(ds.lat.min()) - 0.1, float(ds.lat.max()) + 0.15, 0.05, dtype="float32")
    lon = np.arange(float(ds.lon.min()) - 0.125, float(ds.lon.max()) + 0.1875, 0.0625, dtype="float32")
    geos_input = np.moveaxis(ds["GEOSPM25"].values.astype("float32", copy=False), -1, 0)
    geos = np.moveaxis(model.predict(geos_input, batch_size=batch_size, verbose=0)[:, :, :, 0], 0, -1).astype("float32", copy=False)
    dnn_input = np.moveaxis(ds["BC_DNN_PM25"].values.astype("float32", copy=False), -1, 0)
    dnn = np.moveaxis(model.predict(dnn_input, batch_size=batch_size, verbose=0)[:, :, :, 0], 0, -1).astype("float32", copy=False)
    return xr.Dataset(
        data_vars={
            "DS_GEOSPM25": (["lat", "lon", "time"], geos),
            "DS_BC_DNN_PM25": (["lat", "lon", "time"], dnn),
        },
        coords={"lat": (["lat"], lat), "lon": (["lon"], lon), "time": ds["time"]},
        attrs={"description": "Downscaled PM2.5"},
    )


def crop_output_domain(ds: xr.Dataset, lat_bounds: tuple[float, float], lon_bounds: tuple[float, float]) -> xr.Dataset:
    return ds.sel(lat=coordinate_slice(ds.lat.values, *lat_bounds), lon=coordinate_slice(ds.lon.values, *lon_bounds))


def date_derived_output_name(date_str: str) -> str:
    return f"v1_4_DS_{date_str}.nc"


def date_derived_bc_name(date_str: str) -> str:
    return f"v1_4_BC_{date_str}.nc"


def write_geotiffs(ds: xr.Dataset, output_dir: Path, date_str: str) -> list[Path]:
    import rasterio
    from rasterio.transform import from_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    lon = ds.lon.values
    lat = ds.lat.values
    transform = from_bounds(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()), len(lon), len(lat))
    written: list[Path] = []
    for var_name in ds.data_vars:
        for time_index, time_value in enumerate(ds.time.values):
            timestamp = pd.to_datetime(time_value).strftime("%Y%m%d_%H%M")
            output = output_dir / f"{var_name}_{date_str}_{timestamp}.tif"
            data = ds[var_name].isel(time=time_index).values.astype("float32")
            if lat[0] < lat[-1]:
                data = data[::-1, :]
            with rasterio.open(
                output,
                "w",
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                compress="deflate",
                nodata=-999.0,
            ) as dataset:
                dataset.write(data, 1)
            written.append(output)
    return written


def run_pipeline(config: PipelineConfig, logger: Callable[[str], None] = print) -> PipelineResult:
    validate_date(config.run_date)
    validate_domain(config.lat_min, config.lat_max, config.lon_min, config.lon_max)

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer(logger)
    requested_lat_bounds = (config.lat_min, config.lat_max)
    requested_lon_bounds = (config.lon_min, config.lon_max)
    model_lat_bounds, model_lon_bounds = expand_bounds_for_model(requested_lat_bounds, requested_lon_bounds)

    if model_lat_bounds != requested_lat_bounds or model_lon_bounds != requested_lon_bounds:
        logger(
            "Requested domain is smaller than the model input requirement. "
            f"Fetching padded model domain lat {model_lat_bounds}, lon {model_lon_bounds}, then cropping final output."
        )

    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pass

    cached_input = get_input_file_opendap(config.run_date, model_lat_bounds, model_lon_bounds, timer, logger)
    mx, mn = load_scalars()
    df = prepare_features(cached_input, model_lat_bounds, model_lon_bounds, logger)
    timer.mark("prepared feature dataframe")
    bias_models, ensemble_models, downscale_model = load_models(logger)
    timer.mark("loaded models")
    bc_ds = predict_bias_correction(df, config.run_date, bias_models, ensemble_models, mx, mn, config.batch_size)
    timer.mark("bias correction complete")

    bc_path = PROJECT_ROOT / "OUT_BC" / date_derived_bc_name(config.run_date)
    bc_path.parent.mkdir(parents=True, exist_ok=True)
    bc_ds.to_netcdf(bc_path, mode="w", format="NETCDF4", encoding=NETCDF_ENCODING_BC)
    timer.mark("wrote bias-corrected intermediate")

    ds_output = crop_output_domain(downscale(bc_ds, downscale_model, config.batch_size), requested_lat_bounds, requested_lon_bounds)
    timer.mark("downscaling and crop complete")
    final_netcdf = output_dir / date_derived_output_name(config.run_date)
    ds_output.to_netcdf(final_netcdf, mode="w", format="NETCDF4", encoding=NETCDF_ENCODING_DS)
    timer.mark("wrote final downscaled NetCDF")

    geotiff_dir = None
    geotiff_count = 0
    if config.write_geotiff:
        geotiff_dir = output_dir / "geotiff"
        geotiff_count = len(write_geotiffs(ds_output, geotiff_dir, config.run_date))
        timer.mark("wrote GeoTIFF outputs")

    if not config.keep_intermediate:
        bc_path.unlink(missing_ok=True)
        bc_return = None
    else:
        bc_return = bc_path

    elapsed = time.time() - started
    logger(f"Completed in {elapsed:.1f} seconds.")
    return PipelineResult(final_netcdf, bc_return, cached_input, geotiff_dir, geotiff_count, elapsed)


def load_downscaled_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)
