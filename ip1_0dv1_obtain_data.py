"""
Data acquisition for air pollution and related inputs.
"""

import io
import os
import zipfile

import pandas as pd
import requests
import rasterio

from ip1_0bv1_config import AIR_POLLUTION_URL, SPECIES_MAPPING


def create_descriptive_key(filename):
    """
    Parse filename structure and create descriptive key.

    Filename structure: v720ut3_uifl_4km_1km_rtracv3_*_{gridname}_{species}_ppb_apr_oct_{statistics}.tif
    """
    parts = filename.split("_")

    gridname = None
    species = None
    statistics = None

    for part in parts:
        if part in ["grd01", "grd02"]:
            gridname = part
        elif gridname and part in SPECIES_MAPPING:
            species = part
        elif part.startswith("p") and part[1:].isdigit():
            statistics = part
        elif part == "mean":
            statistics = part

    if statistics and statistics.startswith("p"):
        stat_display = f"p{statistics[1:]}"
    elif statistics == "mean":
        stat_display = "mean"
    else:
        stat_display = statistics if statistics else "unknown"

    return stat_display


def _open_zip_source(zip_source):
    if zip_source is None:
        zip_source = AIR_POLLUTION_URL

    if os.path.exists(zip_source):
        return zipfile.ZipFile(zip_source, "r")

    response = requests.get(zip_source)
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def load_air_pollution_geotiffs(
    zip_source=AIR_POLLUTION_URL,
    species_mapping=SPECIES_MAPPING,
    include_resolutions=None,
    include_species=None,
    verbose=True,
):
    """
    Load air pollution GeoTIFFs from a zip file or URL into a nested dictionary.

    Returns:
        dict with structure: resolution -> species -> statistic
    """
    air_pollution_dict = {}

    include_resolutions_set = None
    if include_resolutions:
        include_resolutions_set = set(include_resolutions)

    include_species_set = None
    if include_species:
        include_species_set = set(include_species)
        include_species_set |= {
            species_mapping.get(code, code) for code in include_species
        }

    with _open_zip_source(zip_source) as z:
        file_list = z.namelist()
        tif_files = [f for f in file_list if f.endswith(".tif") or f.endswith(".tiff")]

        if verbose:
            print(f"Found {len(tif_files)} TIFF files in total")

        for tif_file in tif_files:
            path_parts = tif_file.split("/")
            if len(path_parts) < 3:
                continue

            resolution_dir = path_parts[0]
            species_dir = path_parts[1]
            filename = os.path.splitext(path_parts[2])[0]

            if include_resolutions_set and resolution_dir not in include_resolutions_set:
                continue

            species_name = species_mapping.get(species_dir, species_dir)
            if include_species_set and species_name not in include_species_set:
                continue

            air_pollution_dict.setdefault(resolution_dir, {})
            air_pollution_dict[resolution_dir].setdefault(species_name, {})

            statistic_key = create_descriptive_key(filename)

            with z.open(tif_file) as file_data:
                with rasterio.open(file_data) as src:
                    air_pollution_dict[resolution_dir][species_name][statistic_key] = {
                        "data": src.read(),
                        "transform": src.transform,
                        "crs": src.crs,
                        "meta": src.meta,
                        "bounds": src.bounds,
                        "width": src.width,
                        "height": src.height,
                        "original_filename": filename,
                        "file_path": tif_file,
                    }

    if verbose:
        print(f"\nSuccessfully organized {len(tif_files)} geotiff files into nested dictionary")
        print(f"Resolution levels: {list(air_pollution_dict.keys())}")
        for resolution in air_pollution_dict.keys():
            species_count = len(air_pollution_dict[resolution])
            print(f"\n{resolution}: {species_count} species")
            for species in air_pollution_dict[resolution].keys():
                stat_count = len(air_pollution_dict[resolution][species])
                stats = list(air_pollution_dict[resolution][species].keys())
                print(f"  {species}: {stat_count} statistics ({', '.join(sorted(stats))})")

    return air_pollution_dict


def obtain_hourly_air_quality_data(folder_name, pollutant_name="benz", resolution="1km"):
    """
    Read hourly air quality data from a zip file.
    """
    zip_path = os.path.join(
        "SourceData", folder_name, f"hourly_{pollutant_name}_{resolution}.zip"
    )

    print(f"Reading hourly {pollutant_name} data from {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_filename = f"hourly_{pollutant_name}_{resolution}.csv"
        with z.open(csv_filename) as f:
            hourly_df = pd.read_csv(f)

    print(f"Successfully loaded {len(hourly_df):,} rows of hourly data")
    print(f"Data shape: {hourly_df.shape}")
    print(f"Columns: {list(hourly_df.columns)}")

    return hourly_df
