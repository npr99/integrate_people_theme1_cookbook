"""
Data cleaning, grid creation, and join utilities.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ip1_0cv1_utils import ensure_dir, format_grid_id, normalize_resolution


def generate_grid_id(row, col, resolution="1km"):
    """Generate a standardized grid identifier from row/col indices."""
    return format_grid_id(resolution, row, col)


def create_grid_polygons_from_raster(raster_data, resolution):
    """
    Convert raster cells to vector polygons with unique IDs, including resolution.

    Args:
        raster_data: Dictionary containing raster information.
        resolution: String representing the resolution (e.g., 'uifl_4km' or '1km').

    Returns:
        GeoDataFrame with polygon for each grid cell.
    """
    transform = raster_data["transform"]
    data_array = raster_data["data"][0]
    height, width = data_array.shape
    crs = raster_data["crs"]

    print("Creating vector grid from raster:")
    print(f"  Dimensions: {width} x {height} = {width * height:,} cells")
    print(f"  CRS: {crs}")
    print(f"  Cell size: {abs(transform[0]):.1f} x {abs(transform[4]):.1f} units")

    polygons = []
    grid_ids = []
    row_indices = []
    col_indices = []

    cell_width = abs(transform[0])
    cell_height = abs(transform[4])
    x_origin = transform[2]
    y_origin = transform[5]

    print("Creating polygons for each grid cell...")

    for row in range(height):
        for col in range(width):
            x_center = x_origin + (col + 0.5) * cell_width
            y_center = y_origin - (row + 0.5) * cell_height

            x_left = x_center - cell_width / 2
            x_right = x_center + cell_width / 2
            y_top = y_center + cell_height / 2
            y_bottom = y_center - cell_height / 2

            cell_polygon = Polygon(
                [
                    (x_left, y_bottom),
                    (x_right, y_bottom),
                    (x_right, y_top),
                    (x_left, y_top),
                    (x_left, y_bottom),
                ]
            )

            grid_id = format_grid_id(resolution, row, col)

            polygons.append(cell_polygon)
            grid_ids.append(grid_id)
            row_indices.append(row)
            col_indices.append(col)

    normalized_resolution = normalize_resolution(resolution)
    resolution_label = normalized_resolution.replace("uifl_", "")
    grid_id_column = f"air_grid_id_{resolution_label}"

    grid_gdf = gpd.GeoDataFrame(
        {
            grid_id_column: grid_ids,
            "row_idx": row_indices,
            "col_idx": col_indices,
            "geometry": polygons,
        },
        crs=crs,
    )

    print(f"Successfully created {len(grid_gdf):,} grid polygons")

    return grid_gdf


def add_raster_values_to_grid(grid_gdf, air_pollution_dict, input_resolution):
    """
    Add raster values from air_pollution_dict to the grid GeoDataFrame.
    """
    for resolution, species_data in air_pollution_dict.items():
        if resolution != input_resolution:
            print(
                f"Skipping raster data for {resolution} as it does not match the grid resolution."
            )
            continue

        for species, stats_data in species_data.items():
            for statistic, raster_data in stats_data.items():
                try:
                    data_array = raster_data["data"][0]
                    raster_values = np.array(
                        [
                            data_array[row, col]
                            for row, col in zip(grid_gdf["row_idx"], grid_gdf["col_idx"])
                        ]
                    )
                except IndexError as exc:
                    print(
                        f"Error accessing raster data for {resolution}, {species}, {statistic}: {exc}"
                    )
                    continue

                column_name = f"{resolution}_{species}_{statistic}"
                if len(column_name) > 50:
                    column_name = column_name[:47] + "..."
                grid_gdf[column_name] = raster_values

    print("Successfully added raster values to the grid.")
    return grid_gdf


def grid_gdf_to_csv(grid_gdf, output_path, drop_geometry=True):
    """Save a grid GeoDataFrame to CSV for fast joins."""
    grid_out = grid_gdf.drop(columns=["geometry"]) if drop_geometry else grid_gdf.copy()
    grid_out.to_csv(output_path, index=False)


def write_grid_files(
    grid_1km_gdf,
    grid_4km_gdf,
    output_dir,
    programname,
    to_wgs84=True,
    include_csv=True,
):
    """
    Save grid layers to GPKG and optionally CSV (without geometry).
    """
    ensure_dir(output_dir)

    grid_1km_out = grid_1km_gdf
    grid_4km_out = grid_4km_gdf

    if to_wgs84:
        grid_1km_out = grid_1km_gdf.to_crs(epsg=4326)
        grid_4km_out = grid_4km_gdf.to_crs(epsg=4326)

    grid_4km_filename = f"{programname}_grid_4km.gpkg"
    grid_4km_filepath = os.path.join(output_dir, grid_4km_filename)
    grid_4km_out.to_file(grid_4km_filepath, driver="GPKG")

    grid_1km_filename = f"{programname}_grid_1km.gpkg"
    grid_1km_filepath = os.path.join(output_dir, grid_1km_filename)
    grid_1km_out.to_file(grid_1km_filepath, driver="GPKG")

    if include_csv:
        grid_4km_csv_path = os.path.join(
            output_dir, grid_4km_filename.replace(".gpkg", ".csv")
        )
        grid_1km_csv_path = os.path.join(
            output_dir, grid_1km_filename.replace(".gpkg", ".csv")
        )
        grid_gdf_to_csv(grid_4km_out, grid_4km_csv_path, drop_geometry=True)
        grid_gdf_to_csv(grid_1km_out, grid_1km_csv_path, drop_geometry=True)

    return grid_1km_out, grid_4km_out


def spatial_join_grid_to_hua(
    grid_gdf, hua_gdf, resolution_name, join_type="intersects", include_grid_info=True
):
    """
    Spatially join grid data to HUA points.
    """
    print(f"Performing spatial join between {resolution_name} grid and HUA data...")
    print(f"Grid shape: {grid_gdf.shape}")
    print(f"HUA shape: {hua_gdf.shape}")

    hua_clean = hua_gdf.copy()
    grid_clean = grid_gdf.copy()

    join_columns_to_remove = ["index_right", "index_left"]
    for col in join_columns_to_remove:
        if col in hua_clean.columns:
            hua_clean = hua_clean.drop(col, axis=1)
        if col in grid_clean.columns:
            grid_clean = grid_clean.drop(col, axis=1)

    if grid_clean.crs != hua_clean.crs:
        print(f"Reprojecting grid from {grid_clean.crs} to {hua_clean.crs}")
        grid_gdf_proj = grid_clean.to_crs(hua_clean.crs)
    else:
        grid_gdf_proj = grid_clean.copy()

    hua_with_grid = gpd.sjoin(
        hua_clean, grid_gdf_proj, how="left", predicate=join_type, rsuffix=f"_{resolution_name}"
    )

    if "index_right" in hua_with_grid.columns:
        hua_with_grid = hua_with_grid.rename(
            columns={"index_right": f"{resolution_name}_grid_index"}
        )

    if not include_grid_info:
        grid_cols = set(grid_gdf_proj.columns)
        keep_cols = {"row_idx", "col_idx"}
        dynamic_grid_id_cols = [col for col in grid_cols if col.startswith("air_grid_id_")]
        keep_cols.update(dynamic_grid_id_cols)
        drop_cols = [c for c in grid_cols if c not in keep_cols and c in hua_with_grid.columns]
        if drop_cols:
            hua_with_grid = hua_with_grid.drop(columns=drop_cols)

    print(f"Spatial join complete. Result shape: {hua_with_grid.shape}")
    return hua_with_grid


def add_low_income_renter_variable(df, income_column="randincome", tenure_column="ownershp"):
    """
    Create a Low Income Renter status variable for a dataframe.
    """
    median_income = df[income_column].median()
    print(f"Regional median household income: ${median_income:,.0f}")

    df["Low Income Renter Status"] = "Not Low Income Renter"

    low_income_renter_condition = (
        (df[tenure_column] == 2)
        & (df[income_column] < median_income)
        & (df[income_column].notna())
    )
    df.loc[low_income_renter_condition, "Low Income Renter Status"] = "Low Income Renter"

    missing_data_condition = df[income_column].isna() | df[tenure_column].isna()
    df.loc[missing_data_condition, "Low Income Renter Status"] = np.nan

    print("\nLow Income Renter Status distribution:")
    print(df["Low Income Renter Status"].value_counts(dropna=False))

    low_income_renters = df[df["Low Income Renter Status"] == "Low Income Renter"]
    if len(low_income_renters) > 0:
        print("\nLow-income renter households:")
        print(f"  Count: {len(low_income_renters):,}")
        print(f"  Mean income: ${low_income_renters[income_column].mean():,.0f}")
        print(f"  Median income: ${low_income_renters[income_column].median():,.0f}")
        print(
            f"  Income range: ${low_income_renters[income_column].min():,.0f} - ${low_income_renters[income_column].max():,.0f}"
        )

    return df
