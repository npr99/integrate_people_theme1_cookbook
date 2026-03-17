# integrate_people_theme1_cookbook

This project integrates hazardous air pollutant (HAP) model outputs with housing/population data for Southeast Texas, then generates hotspot layers and interactive web maps.

## What the Python files do (`.py`)

Core processing modules:

- `ip1_0bv1_config.py`  
	Shared constants and configuration (species names, URLs, contour defaults, demographic thresholds, color mappings).

- `ip1_0cv1_utils.py`  
	Utility helpers for resolution naming, grid IDs, and directory creation.

- `ip1_0dv1_obtain_data.py`  
	Data acquisition functions (download/read pollutant GeoTIFF archives and hourly pollutant zip files).

- `ip1_0ev1_clean_data.py`  
	Cleaning/join pipeline utilities (convert raster cells to vector grids, add raster values to grids, spatially join grid attributes to housing unit points, write outputs).

- `ip1_0fv1_explore_visualize.py`  
	Visualization and mapping utilities (extent plots, raster/vector checks, contour helper functions, map-ready plotting routines).

- `ip1_0gv1_model_statistics.py`  
	Statistical comparison tools (ANOVA-style comparisons, effect size, bootstrap confidence intervals, resolution/group comparison summaries).

- `ip1_0hv1_hotspot_analysis.py`  
	Main hotspot analysis functions (demographic filtering, contour creation, weighted exposure hotspot construction, interactive Folium map creation, output save helpers).

Pipeline helpers:

- `ip1_1av1_common.py`  
	Convenience import layer that re-exports the major functions from the modules above for notebook use.

- `update_hotspot_analysis.py`  
	One-time maintenance script used to patch hotspot function signatures/docstrings (development utility, not part of routine analysis runs).

## What the notebooks do (`.ipynb`)

Primary top-level notebooks:

- `ip1_2bv2_airdata_2026-02-23.ipynb`  
	Converts modeled HAP GeoTIFF datasets into tabular/grid outputs (CSV and GeoPackage) for 1 km and 4 km processing.

- `ip1_2cv2_popair_2026-02-27.ipynb`  
	Merges housing unit allocation/population attributes with air pollutant grid values through spatial joins.

- `ip1_2dv1_hotspotspopair_2026-03-05.ipynb`  
	Builds hotspot contours and comparison layers from merged population-air data.

- `ip1_2dv2_hotspotspopair_2026-03-05.ipynb`  
	Produces map-ready hotspot outputs for selected pollutants (for example Benzene and Ethylene Oxide at multiple resolutions).

- `ip1_2dv3_hotspotspopair_2026-03-06.ipynb`  
	Extended hotspot run including optional group-quarters populations (for example correctional, juvenile, and nursing facilities) and additional checks.

- `ip1_3av1_hotspotspopair_2026-03-03.ipynb`  
	Exploratory notebook for inspecting and visualizing hotspot/population-air outputs.

## What the HTML outputs represent (`.html`)

The HTML files are interactive web maps generated from the hotspot workflow (typically via Folium). They are intended for browser-based review and sharing.

- Files in `InteractiveMaps/` are publish-ready interactive maps.
- A typical map HTML includes pollutant-linked hotspot polygons, demographic comparison layers, and toggles/controls to inspect where population exposure is elevated.
- Example outputs currently include:
	- `InteractiveMaps/ip1_2dv2_hotspotspopair_uifl_1km_Benzene_mean.html`
	- `InteractiveMaps/ip1_2dv2_hotspotspopair_uifl_1km_Ethylene Oxide_mean.html`

Supporting output folders (for example `ip1_2dv1_hotspotspopair/`, `ip1_2dv2_hotspotspopair/`, `ip1_2dv3_hotspotspopair/`) store intermediate and final geospatial artifacts used to build those HTML maps.

## Publish maps with GitHub Pages (manual branch updates)

This repository can be published as a static website without automation. The website updates only when you push changes to your selected Pages branch.

1. In GitHub, go to **Settings > Pages**.
2. Set **Source** to **Deploy from a branch**.
3. Select your publish branch and folder (`/docs`).
4. Keep `docs/index.html` as the site entry page.
5. Keep map HTML files in `docs/InteractiveMaps/`.
6. Update the `MAP_OPTIONS` list in `docs/index.html` to add/remove HAP choices shown in the left selector.
7. Commit and push updates to the publish branch whenever you want the website to refresh.

The published `docs/index.html` is a single-page viewer: select a HAP on the left, and the map loads in an embedded frame on the right.

Example site URL pattern:

`https://npr99.github.io/integrate_people_theme1_cookbook/`
