# AQ Downscaling v2 GUI

AQ Downscaling v2 is a Streamlit application for running GEOS-FP PM2.5 bias correction and 5 km downscaling from a graphical interface.

The app supports:

- Forecast date and domain selection
- Cairo 2 degree by 2 degree default domain
- Custom latitude and longitude bounds
- OPeNDAP-based GEOS-FP subsetting for smaller downloads
- NetCDF output
- Optional GeoTIFF output
- Optional intermediate 25 km bias-corrected NetCDF output
- Interactive PM2.5 map viewer with basemaps, opacity control, timestep slider, previous/next, reset, and play/pause

## Repository

```text
https://github.com/alqamahsayeed12/AQ_Downscaling_v2
```

## 1. Clone The Project

```bash
git clone https://github.com/alqamahsayeed12/AQ_Downscaling_v2.git
cd AQ_Downscaling_v2
```

If the repository is private, GitHub will require authentication. Use a GitHub personal access token as the password when prompted. GitHub account passwords do not work for Git pushes or private HTTPS access.

## 2. Confirm Required Files

After cloning, the project should contain:

```text
app.py
aq_downscaling_app/
Model/
Scalars/
downloadGEOSDataParams.json
environment_cli.yml
README.md
```

Important model and scalar assets:

```text
Model/model_downscale_v1.h5
Model/v1_4_day1_dnn_bias_Correction_ensemble.h5
Model/v1_4_day2_dnn_bias_Correction_ensemble.h5
Model/v1_4_day3_dnn_bias_Correction_ensemble.h5
Model/v1_4_dnn_bias_Correction_day*_fold*.h5
Scalars/max_min4.csv
```

## 3. Create The Conda Environment

Install Miniconda or Anaconda first if `conda` is not available.

Create the environment:

```bash
conda env create -f environment_cli.yml
```

Activate it:

```bash
conda activate aq_downscaling
```

If the environment already exists and you need to update it:

```bash
conda env update -f environment_cli.yml --prune
conda activate aq_downscaling
```

## 4. Run The App

From the project folder:

```bash
streamlit run app.py
```

The app normally opens at:

```text
http://localhost:8501
```

If the browser does not open automatically, copy that URL into your browser.

## 5. Process A Forecast

In the left sidebar:

1. Select the forecast initialization date.
2. Choose the domain.
3. Use `Cairo 2° x 2°` for the default Cairo domain, or choose `Custom` and enter your own bounds.
4. Select the output folder. The default is:

```text
./OUT_DS
```

5. Keep `Write GeoTIFFs` enabled if you want GeoTIFF output.
6. Keep `Keep 25 km bias-corrected NetCDF` enabled if you want the intermediate bias-corrected file.
7. Click `Run Downscaling`.

Progress messages appear in the app while the workflow downloads GEOS-FP subsets, applies bias correction, runs the downscaling model, and writes outputs.

## 6. View The Forecast Map

After processing completes, the right panel displays the 5 km bias-corrected PM2.5 forecast.

Map controls include:

- Basemap selector
- PM2.5 opacity slider
- Forecast timestep slider
- Previous timestep button
- Next timestep button
- Reset button
- Play/Pause animation button
- PM2.5 colorbar
- Min, mean, and max PM2.5 values for the active timestep

The map viewer runs timestep playback in the browser, so moving between frames does not rerun the full Streamlit app.

## 7. Output Files

Final downscaled NetCDF:

```text
OUT_DS/v1_4_DS_YYYYMMDD.nc
```

GeoTIFF outputs, if enabled:

```text
OUT_DS/geotiff/*.tif
```

Intermediate 25 km bias-corrected NetCDF, if enabled:

```text
OUT_BC/v1_4_BC_YYYYMMDD.nc
```

Downloaded and subset GEOS-FP files are cached in:

```text
IN_NetCDF/
```

These output/cache folders are ignored by Git because they are generated files.

## 8. Recommended Test Run

For a quick test, use:

```text
Date: yesterday's date
Preset: Cairo 2° x 2°
Write GeoTIFFs: enabled
Keep 25 km bias-corrected NetCDF: enabled
Output folder: ./OUT_DS
```

Then click `Run Downscaling`.

## 9. Troubleshooting

If `streamlit` is not found:

```bash
conda activate aq_downscaling
streamlit run app.py
```

If the app cannot import TensorFlow or Keras, recreate the environment:

```bash
conda env remove -n aq_downscaling
conda env create -f environment_cli.yml
conda activate aq_downscaling
```

If downloads fail, rerun the workflow. Network interruptions can happen during GEOS-FP data access.

If port `8501` is already in use:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

If GitHub asks for a password during `git push`, paste a GitHub personal access token instead of your GitHub account password.

## 10. Project Layout

```text
AQ_Downscaling_v2/
├── app.py
├── aq_downscaling_app/
│   ├── __init__.py
│   └── pipeline.py
├── Model/
├── Scalars/
├── downloadGEOSDataParams.json
├── environment_cli.yml
├── README.md
├── IN_NetCDF/   # generated cache, ignored by Git
├── OUT_BC/      # generated intermediate output, ignored by Git
└── OUT_DS/      # generated final output, ignored by Git
```

## 11. Notes

- The app is intended to run locally or on a cloud/server machine.
- GitHub stores the code but does not run the interactive Streamlit app directly.
- For remote access, deploy the project on a VM or server and run Streamlit with:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
