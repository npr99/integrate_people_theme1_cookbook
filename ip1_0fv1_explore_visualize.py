"""
Exploratory visualization and mapping utilities.
"""

import base64
import io
import os
from datetime import datetime

import contextily as ctx
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sg
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from rasterio.plot import show
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ip1_0bv1_config import TEXAS_CITIES


def create_data_extent_box(data_dict):
    """Create a bounding box polygon from raster data bounds."""
    bounds = data_dict["bounds"]
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return bbox, bounds


def plot_data_extent_on_basemap(
    air_pollution_dict, resolution="uifl_1km", species="Benzene", statistic="p25"
):
    """Plot data extent as polygon on contextily base map."""
    data_dict = air_pollution_dict[resolution][species][statistic]
    bbox_geom, bounds = create_data_extent_box(data_dict)

    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=data_dict["crs"])
    bbox_wgs84 = bbox_gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(12, 10))

    bbox_wgs84.boundary.plot(ax=ax, color="red", linewidth=3, label="Data Extent")
    bbox_wgs84.plot(ax=ax, alpha=0.2, color="red", edgecolor="red", linewidth=3)

    try:
        ctx.add_basemap(
            ax, crs=bbox_wgs84.crs.to_string(), source=ctx.providers.Esri.WorldStreetMap
        )
        basemap_source = "Esri WorldStreetMap"
    except Exception:
        try:
            ctx.add_basemap(
                ax,
                crs=bbox_wgs84.crs.to_string(),
                source=ctx.providers.OpenStreetMap.Mapnik,
            )
            basemap_source = "OpenStreetMap"
        except Exception:
            print("Warning: Could not add base map")
            basemap_source = "None"

    cities_data = []
    for city_name, coords in TEXAS_CITIES.items():
        cities_data.append({"city": city_name, "geometry": Point(coords["lon"], coords["lat"])})

    cities_gdf = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")
    cities_gdf.plot(
        ax=ax, color="blue", markersize=100, alpha=0.8, edgecolor="white", linewidth=2, zorder=10
    )

    for _, row in cities_gdf.iterrows():
        ax.annotate(
            row["city"],
            (row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            zorder=11,
        )

    ax.set_title(
        f"Air Pollution Data Extent\n{species} {statistic} ({resolution})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")

    date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    fig.text(
        0.5,
        0.00,
        "Data extent layer from sensor network",
        ha="center",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    print("Data Extent Information:")
    print(f"  Resolution: {resolution}")
    print(f"  Species: {species}")
    print(f"  Statistic: {statistic}")
    print(f"  CRS: {data_dict['crs']}")
    print("  Bounds (original CRS):")
    print(f"    West:  {bounds.left:.6f}")
    print(f"    East:  {bounds.right:.6f}")
    print(f"    South: {bounds.bottom:.6f}")
    print(f"    North: {bounds.top:.6f}")
    print(f"  Area: {bbox_geom.area:.2e} square units")

    plt.tight_layout()
    plt.show()

    return bbox_gdf


def plot_multiple_extents_comparison(air_pollution_dict):
    """Compare the extents of 1km and 4km resolution data."""
    fig, ax = plt.subplots(figsize=(14, 10))

    data_1km = air_pollution_dict["uifl_1km"]["Benzene"]["p25"]
    data_4km = air_pollution_dict["uifl_4km"]["Benzene"]["p25"]

    bbox_1km, _ = create_data_extent_box(data_1km)
    bbox_4km, _ = create_data_extent_box(data_4km)

    bbox_1km_gdf = gpd.GeoDataFrame([1], geometry=[bbox_1km], crs=data_1km["crs"])
    bbox_4km_gdf = gpd.GeoDataFrame([1], geometry=[bbox_4km], crs=data_4km["crs"])

    bbox_1km_wgs = bbox_1km_gdf.to_crs(epsg=4326)
    bbox_4km_wgs = bbox_4km_gdf.to_crs(epsg=4326)

    bbox_1km_wgs.boundary.plot(ax=ax, color="red", linewidth=3, label="1km Resolution")
    bbox_1km_wgs.plot(ax=ax, alpha=0.2, color="red", edgecolor="red", linewidth=3)

    bbox_4km_wgs.boundary.plot(
        ax=ax, color="blue", linewidth=3, label="4km Resolution", linestyle="--"
    )
    bbox_4km_wgs.plot(ax=ax, alpha=0.1, color="blue", edgecolor="blue", linewidth=3)

    try:
        ctx.add_basemap(ax, crs=bbox_1km_wgs.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    except Exception:
        print("Warning: Could not add base map")

    cities_data = []
    for city_name, coords in TEXAS_CITIES.items():
        cities_data.append({"city": city_name, "geometry": Point(coords["lon"], coords["lat"])})

    cities_gdf = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")
    cities_gdf.plot(
        ax=ax, color="darkgreen", markersize=80, alpha=0.9, edgecolor="white", linewidth=2, zorder=10
    )

    for _, row in cities_gdf.iterrows():
        ax.annotate(
            row["city"],
            (row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            zorder=11,
        )

    ax.set_title(
        "Air Pollution Data Coverage Comparison\nSoutheast Texas Urban Integrated Field Lab",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=12)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    plt.tight_layout()
    plt.show()

    print("Data Coverage Comparison:")
    print(f"  1km Resolution Area: {bbox_1km.area:.2e} square units")
    print(f"  4km Resolution Area: {bbox_4km.area:.2e} square units")
    print(f"  Coverage Overlap: {'Yes' if bbox_1km.intersects(bbox_4km) else 'No'}")


def plot_raster_values_on_grid(grid_gdf, column_name):
    """Plot raster values on the vector grid with CartoDB Positron basemap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    if grid_gdf.crs.to_string() != "EPSG:4326":
        grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
    else:
        grid_wgs84 = grid_gdf

    grid_wgs84.plot(
        column=column_name,
        ax=ax,
        cmap="viridis",
        legend=True,
        legend_kwds={"label": f"{column_name} Concentration (ppb)"},
        edgecolor="black",
        linewidth=0.1,
        alpha=0.8,
    )

    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=grid_wgs84.crs.to_string())
    except Exception as exc:
        print(f"Warning: Could not add basemap: {exc}")

    ax.set_title(
        f"Air Pollution Raster Values on Vector Grid\n{column_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    plt.tight_layout()
    plt.show()


def compare_geotiff_vs_vector(
    air_pollution_dict, grid_gdf, resolution="uifl_4km", species="Benzene", statistic="p100"
):
    """Create side-by-side comparison of original GeoTIFF and vector file with raster values."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    raster_data = air_pollution_dict[resolution][species][statistic]
    data_array = raster_data["data"][0]
    transform = raster_data["transform"]

    column_name = f"{resolution}_{species}_{statistic}"

    ax1 = axes[0]

    if str(raster_data["crs"]) != "EPSG:4326":
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            raster_data["crs"], dst_crs, raster_data["width"], raster_data["height"], *raster_data["bounds"]
        )
        data_array_wgs84 = np.empty((height, width), dtype=data_array.dtype)

        reproject(
            source=data_array,
            destination=data_array_wgs84,
            src_transform=raster_data["transform"],
            src_crs=raster_data["crs"],
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        show(data_array_wgs84, transform=transform, ax=ax1, cmap="viridis")
    else:
        show(data_array, transform=transform, ax=ax1, cmap="viridis")

    try:
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron, crs="EPSG:4326", alpha=0.7)
    except Exception as exc:
        print(f"Could not add basemap to raster plot: {exc}")

    ax1.set_title(
        f"Original GeoTIFF\n{species} {statistic} ({resolution})", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Longitude", fontsize=12)
    ax1.set_ylabel("Latitude", fontsize=12)

    ax2 = axes[1]

    if grid_gdf.crs.to_string() != "EPSG:4326":
        grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
    else:
        grid_wgs84 = grid_gdf

    grid_wgs84.plot(
        column=column_name,
        ax=ax2,
        cmap="viridis",
        legend=True,
        legend_kwds={"label": f"{species} {statistic} (ppb)", "shrink": 0.8},
        edgecolor="black",
        linewidth=0.05,
        alpha=0.8,
    )

    try:
        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron, crs=grid_wgs84.crs.to_string(), alpha=0.7)
    except Exception as exc:
        print(f"Warning: Could not add basemap to vector plot: {exc}")

    ax2.set_title(
        f"Vector Grid with Raster Values\n{species} {statistic} ({resolution})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Longitude", fontsize=12)
    ax2.set_ylabel("Latitude", fontsize=12)

    fig.suptitle(
        f"GeoTIFF vs Vector Comparison: {species} {statistic} ({resolution})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.show()

    print("\nComparison Statistics:")
    print("Original GeoTIFF:")
    print(f"  Data range: {data_array.min():.3f} to {data_array.max():.3f}")
    print(f"  Mean value: {data_array.mean():.3f}")
    print(f"  Shape: {data_array.shape}")

    print("\nVector Grid:")
    vector_values = grid_gdf[column_name]
    print(f"  Data range: {vector_values.min():.3f} to {vector_values.max():.3f}")
    print(f"  Mean value: {vector_values.mean():.3f}")
    print(f"  Number of cells: {len(vector_values)}")

    raster_flat = data_array.flatten()
    vector_flat = vector_values.values
    if len(raster_flat) == len(vector_flat):
        differences = abs(raster_flat - vector_flat)
        max_diff = differences.max()
        print("\nData Validation:")
        print(f"  Maximum difference between raster and vector: {max_diff:.6f}")
        if max_diff < 1e-10:
            print("  ✓ Values match perfectly between raster and vector!")
        else:
            print(f"  ⚠ Some differences found (max: {max_diff:.6f})")
    else:
        print(f"\nData shapes differ: Raster {raster_flat.shape} vs Vector {vector_flat.shape}")


def plot_vector_grids_comparison(grid_4km_gdf, grid_1km_gdf, programname, output_dir):
    """Compare the vector polygon grids of 1km and 4km resolution data."""
    fig, ax = plt.subplots(figsize=(16, 12))

    grid_4km_wgs84 = grid_4km_gdf.to_crs("EPSG:4326")

    try:
        if grid_1km_gdf is not None and len(grid_1km_gdf) > 0:
            grid_1km_wgs84 = grid_1km_gdf.to_crs("EPSG:4326")
            print(f"Plotting full 1km grid with {len(grid_1km_wgs84):,} grid cells")
        else:
            grid_1km_wgs84 = None
            print("1km grid not available - plotting 4km grid only")
    except (TypeError, AttributeError):
        grid_1km_wgs84 = None
        print("1km grid not created yet - plotting 4km grid only")

    bounds_4km = grid_4km_wgs84.total_bounds
    margin = 0.1
    x_margin = (bounds_4km[2] - bounds_4km[0]) * margin
    y_margin = (bounds_4km[3] - bounds_4km[1]) * margin

    ax.set_xlim(bounds_4km[0] - x_margin, bounds_4km[2] + x_margin)
    ax.set_ylim(bounds_4km[1] - y_margin, bounds_4km[3] + y_margin)

    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldStreetMap, alpha=0.6)
    except Exception as exc:
        print(f"Warning: Could not add base map: {exc}")

    grid_4km_wgs84.boundary.plot(
        ax=ax, color="blue", linewidth=1.5, label="4km Resolution Grid", alpha=0.8
    )

    if grid_1km_wgs84 is not None:
        grid_1km_wgs84.boundary.plot(
            ax=ax, color="red", linewidth=0.3, label="1km Resolution Grid", alpha=0.7
        )

    cities_data = []
    for city_name, coords in TEXAS_CITIES.items():
        cities_data.append({"city": city_name, "geometry": Point(coords["lon"], coords["lat"])})

    cities_gdf = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")
    cities_gdf.plot(
        ax=ax, color="darkgreen", markersize=120, alpha=0.9, edgecolor="white", linewidth=3, zorder=15
    )

    for _, row in cities_gdf.iterrows():
        ax.annotate(
            row["city"],
            (row.geometry.x, row.geometry.y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="darkgreen"),
            zorder=16,
        )

    ax.set_title(
        "Air Pollution Vector Grid Comparison\nSoutheast Texas Urban Integrated Field Lab",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    legend_elements = [
        plt.Line2D([0], [0], color="blue", linewidth=2, label="4km Resolution Grid"),
    ]

    if grid_1km_wgs84 is not None:
        legend_elements.append(plt.Line2D([0], [0], color="red", linewidth=1, label="1km Resolution Grid"))

    date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    fig.text(0.2, -0.001, f"Provenance: {programname} {date}", ha="left", va="bottom", fontsize=8, color="gray")

    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    plt.tight_layout()

    if output_dir and os.path.exists(output_dir):
        map_filename = f"{programname}_vector_grids_comparison.png"
        map_filepath = os.path.join(output_dir, map_filename)
        plt.savefig(map_filepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n✓ Saved comparison map: {map_filepath}")

    plt.show()

    print("\nVector Grid Comparison Summary:")
    print(f"  4km Grid: {len(grid_4km_gdf):,} total polygons")
    try:
        if grid_1km_gdf is not None and len(grid_1km_gdf) > 0:
            print(f"  1km Grid: {len(grid_1km_gdf):,} total polygons")
            print(
                f"  Resolution Ratio: {len(grid_1km_gdf) / len(grid_4km_gdf):.1f}x more cells in 1km"
            )

            area_4km = grid_4km_gdf.to_crs("EPSG:3857").area.mean() / 1_000_000
            area_1km = grid_1km_gdf.to_crs("EPSG:3857").area.mean() / 1_000_000
            print(f"  Average cell area - 4km: {area_4km:.1f} km², 1km: {area_1km:.1f} km²")
        else:
            print("  1km Grid: Not created yet")
    except (TypeError, AttributeError):
        print("  1km Grid: Variable not defined")


def plot_vector_grids_comparison_zoomed(
    grid_4km_gdf, grid_1km_gdf, programname, output_dir, overlay_tfsites=False, tfsites_gdf=None
):
    """Create zoomed-in comparison map focusing on 1km grid extent."""
    fig, ax = plt.subplots(figsize=(18, 14))

    grid_4km_wgs84 = grid_4km_gdf.to_crs("EPSG:4326")

    grid_1km_wgs84 = None
    zoom_bounds = None

    try:
        if grid_1km_gdf is not None and len(grid_1km_gdf) > 0:
            grid_1km_wgs84 = grid_1km_gdf.to_crs("EPSG:4326")
            zoom_bounds = grid_1km_wgs84.total_bounds
            print(f"Zooming to 1km grid extent with {len(grid_1km_wgs84):,} grid cells")
        else:
            zoom_bounds = grid_4km_wgs84.total_bounds
            print("1km grid not available - using 4km grid extent")
    except (TypeError, AttributeError):
        zoom_bounds = grid_4km_wgs84.total_bounds
        print("1km grid not created yet - using 4km grid extent")

    margin = 0.05
    x_margin = (zoom_bounds[2] - zoom_bounds[0]) * margin
    y_margin = (zoom_bounds[3] - zoom_bounds[1]) * margin

    ax.set_xlim(zoom_bounds[0] - x_margin, zoom_bounds[2] + x_margin)
    ax.set_ylim(zoom_bounds[1] - y_margin, zoom_bounds[3] + y_margin)

    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldStreetMap, alpha=0.8)
        print("Using Esri WorldStreetMap basemap")
    except Exception:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, alpha=0.8)
            print("Using satellite imagery basemap")
        except Exception:
            try:
                ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
                print("Using OpenStreetMap basemap")
            except Exception as exc:
                print(f"Warning: Could not add base map: {exc}")

    grid_4km_wgs84.boundary.plot(ax=ax, color="blue", linewidth=2.0, label="4km Resolution Grid", alpha=0.9)

    if grid_1km_wgs84 is not None:
        grid_1km_wgs84.boundary.plot(ax=ax, color="red", linewidth=0.5, label="1km Resolution Grid", alpha=0.8)

    cities_data = []
    for city_name, coords in TEXAS_CITIES.items():
        cities_data.append({"city": city_name, "geometry": Point(coords["lon"], coords["lat"])})

    cities_gdf = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")
    cities_gdf.plot(
        ax=ax, color="darkgreen", markersize=120, alpha=0.9, edgecolor="white", linewidth=3, zorder=15
    )

    for _, row in cities_gdf.iterrows():
        ax.annotate(
            row["city"],
            (row.geometry.x, row.geometry.y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="darkgreen"),
            zorder=16,
        )

    if grid_1km_wgs84 is not None:
        title_text = (
            f"Detailed Vector Grid Comparison (Zoomed to 1km Extent)\n"
            f"Southeast Texas - {len(grid_1km_gdf):,} cells at 1km, {len(grid_4km_gdf):,} cells at 4km"
        )
    else:
        title_text = "Vector Grid Comparison (4km Resolution)\nSoutheast Texas Urban Integrated Field Lab"

    ax.set_title(title_text, fontsize=16, fontweight="bold", pad=25)

    legend_elements = [
        plt.Line2D([0], [0], color="blue", linewidth=3, label="4km Resolution Grid"),
    ]

    if grid_1km_wgs84 is not None:
        legend_elements.append(plt.Line2D([0], [0], color="red", linewidth=2, label="1km Resolution Grid"))

    if overlay_tfsites and tfsites_gdf is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="*", color="yellow", label="Task Force Sites", markeredgecolor="black", linestyle="None")
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.95, fancybox=True, shadow=True)

    ax.set_xlabel("Longitude (°W)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=11)

    date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    fig.text(0.5, -0.01, f"Provenance: {programname} {date}", ha="left", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()

    if output_dir and os.path.exists(output_dir):
        zoom_map_filename = f"{programname}_vector_grids_comparison_zoomed.png"
        zoom_map_filepath = os.path.join(output_dir, zoom_map_filename)
        plt.savefig(zoom_map_filepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n✓ Saved zoomed comparison map: {zoom_map_filepath}")

    plt.show()

    print("\n" + "=" * 70)
    print("ZOOMED VECTOR GRID COMPARISON")
    print("=" * 70)
    print(
        f"Zoom extent: {zoom_bounds[0]:.4f}°W to {zoom_bounds[2]:.4f}°W, "
        f"{zoom_bounds[1]:.4f}°N to {zoom_bounds[3]:.4f}°N"
    )
    print(f"Map area: ~{(zoom_bounds[2] - zoom_bounds[0]) * 111:.0f} km × {(zoom_bounds[3] - zoom_bounds[1]) * 111:.0f} km")

    print("\nGrid Details:")
    print(f"  4km Grid: {len(grid_4km_gdf):,} total polygons")

    try:
        if grid_1km_gdf is not None and len(grid_1km_gdf) > 0:
            print(f"  1km Grid: {len(grid_1km_gdf):,} total polygons")
            print(f"  Resolution Factor: {len(grid_1km_gdf) / len(grid_4km_gdf):.1f}x more detail in 1km")

            area_km2 = (zoom_bounds[2] - zoom_bounds[0]) * (zoom_bounds[3] - zoom_bounds[1]) * 111 * 111
            density_4km = len(grid_4km_gdf) / area_km2
            density_1km = len(grid_1km_gdf) / area_km2
            print(f"  Grid Density: 4km = {density_4km:.2f} cells/km², 1km = {density_1km:.2f} cells/km²")
        else:
            print("  1km Grid: Not available")
    except (TypeError, AttributeError):
        print("  1km Grid: Not created")


def plot_site_zoom(
    site_name,
    grid_4km_gdf,
    programname,
    output_dir,
    buffer_km=5,
    overlay_tfsites=True,
    tfsites_gdf=None,
    save=True,
):
    """Create a zoomed comparison map centered on a named study site."""
    if tfsites_gdf is None or len(tfsites_gdf) == 0:
        print("tfsites_gdf not provided or empty. Cannot locate study site.")
        return None, None

    name_cols = [c for c in tfsites_gdf.columns if str(c).lower() in ("name", "site", "site_name", "sitename", "title")]
    match_col = None
    if name_cols:
        match_col = name_cols[0]
    else:
        for c in tfsites_gdf.columns:
            if tfsites_gdf[c].dtype == object:
                match_col = c
                break

    if match_col is None:
        print("No suitable name column found in tfsites_gdf; cannot match site_name")
        return None, None

    matches = tfsites_gdf[tfsites_gdf[match_col].str.lower() == site_name.lower()]
    if len(matches) == 0:
        matches = tfsites_gdf[tfsites_gdf[match_col].str.lower().str.contains(site_name.lower(), na=False)]
    if len(matches) == 0:
        print(f"No study site matching '{site_name}' found in {match_col}")
        return None, None

    site_row = matches.iloc[0]
    site_geom = site_row.geometry

    try:
        site_proj = gpd.GeoSeries([site_geom], crs=tfsites_gdf.crs).to_crs(epsg=3857).iloc[0]
    except Exception:
        site_proj = gpd.GeoSeries([site_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    buffer_m = buffer_km * 1000.0
    buffer_proj = site_proj.buffer(buffer_m)

    buffer_wgs = gpd.GeoSeries([buffer_proj], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    zoom_bounds = buffer_wgs.bounds

    fig, ax = plt.subplots(figsize=(14, 12))

    minx, miny, maxx, maxy = zoom_bounds
    xpad = (maxx - minx) * 0.08
    ypad = (maxy - miny) * 0.08
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)

    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldStreetMap, alpha=0.8)
    except Exception:
        print("Warning: Could not add base map.")

    try:
        g4 = grid_4km_gdf.to_crs(epsg=4326)
        g4.boundary.plot(ax=ax, color="blue", linewidth=1.2, label="4km grid")
    except Exception as exc:
        print(f"Could not plot 4km grid: {exc}")

    gpd.GeoSeries([buffer_wgs], crs="EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.0, linestyle="--")

    if overlay_tfsites and tfsites_gdf is not None:
        try:
            tfs = tfsites_gdf.to_crs(epsg=4326)
            tfs.plot(ax=ax, color="yellow", markersize=70, marker="*", edgecolor="black", zorder=20, label="Task force sites", alpha=0.5)
            sel = gpd.GeoDataFrame([site_row], crs=tfsites_gdf.crs).to_crs(epsg=4326)
            sel.plot(
                ax=ax,
                color="orange",
                markersize=150,
                marker="*",
                edgecolor="black",
                zorder=21,
                label=f"Selected: {site_row[match_col]}",
                alpha=0.3,
            )
            for _, row in sel.iterrows():
                if hasattr(row.geometry, "x"):
                    x, y = row.geometry.x, row.geometry.y
                else:
                    centroid = row.geometry.centroid
                    x, y = centroid.x, centroid.y
                ax.annotate(str(row[match_col]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")
        except Exception as exc:
            print(f"Could not overlay tfsites_gdf: {exc}")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Site zoom: {site_row[match_col]} (±{buffer_km} km)")

    ax.legend(loc="upper right")
    date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    fig.text(0.5, 0.01, f"Provenance: {programname} {date}", ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()

    if save and output_dir and os.path.exists(output_dir):
        safe_name = str(site_row[match_col]).replace(" ", "_")
        fname = f"{programname}_sitezoom_{safe_name}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved site zoom map: {fpath}")

    plt.show()

    return fig, ax


def build_people_raster(people_gdf, value_col="numprec", cell_size_m=200, window_radius_m=250):
    """Create a raster from point GeoDataFrame with moving window sum of population."""
    if people_gdf.empty:
        raise ValueError("Input GeoDataFrame is empty after filtering.")
    if people_gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS.")
    if value_col not in people_gdf.columns:
        raise ValueError(f"Missing value column: {value_col}")

    working_gdf = people_gdf.copy()
    if working_gdf.geometry.isna().all():
        raise ValueError("Input GeoDataFrame has no valid geometries.")
    if working_gdf.crs.is_geographic:
        working_gdf = working_gdf.to_crs(working_gdf.estimate_utm_crs())

    minx, miny, maxx, maxy = working_gdf.total_bounds
    pad = window_radius_m
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    width = int(np.ceil((maxx - minx) / cell_size_m))
    height = int(np.ceil((maxy - miny) / cell_size_m))
    transform = from_origin(minx, maxy, cell_size_m, cell_size_m)

    from rasterio.features import rasterize
    from rasterio.enums import MergeAlg
    from scipy.ndimage import convolve

    shapes = (
        (geom, float(val))
        for geom, val in zip(working_gdf.geometry, working_gdf[value_col].fillna(0))
        if geom is not None
    )
    base_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="float32",
        merge_alg=MergeAlg.add,
    )

    radius_cells = int(np.ceil(window_radius_m / cell_size_m))
    y, x = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    kernel = (x * x + y * y) <= (window_radius_m / cell_size_m) ** 2
    kernel = kernel.astype("float32")

    moving_sum = convolve(base_raster, kernel, mode="constant", cval=0.0)

    dst_crs = "EPSG:4326"
    dst_transform, dst_width, dst_height = calculate_default_transform(
        working_gdf.crs, dst_crs, width, height, minx, miny, maxx, maxy
    )
    moving_sum_ll = np.zeros((dst_height, dst_width), dtype="float32")
    reproject(
        source=moving_sum,
        destination=moving_sum_ll,
        src_transform=transform,
        src_crs=working_gdf.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    return {
        "raster": moving_sum_ll,
        "gdf": working_gdf,
        "dst_crs": dst_crs,
        "dst_transform": dst_transform,
    }


def _get_raster_bounds(raster_result):
    """Extract geographic bounds from a raster result dictionary."""
    raster = raster_result["raster"]
    dst_transform = raster_result["dst_transform"]
    height, width = raster.shape
    left = dst_transform.c
    top = dst_transform.f
    right = left + dst_transform.a * width
    bottom = top + dst_transform.e * height

    if left > right:
        left, right = right, left
    if bottom > top:
        bottom, top = top, bottom

    lat_limit = 85.0511
    bottom = float(np.clip(bottom, -lat_limit, lat_limit))
    top = float(np.clip(top, -lat_limit, lat_limit))
    left = float(np.clip(left, -180.0, 180.0))
    right = float(np.clip(right, -180.0, 180.0))
    if bottom >= top or left >= right:
        epsilon = 1e-4
        if bottom >= top:
            bottom -= epsilon
            top += epsilon
        if left >= right:
            left -= epsilon
            right += epsilon

    return left, bottom, right, top


def plot_people_raster(
    raster_result,
    title="Moving Window Sum of People per Cell",
    basemap=None,
    zoom=11,
    cmap_name="Reds",
    alpha=0.85,
    ax=None,
):
    """Plot a people density raster with basemap and colorbar."""
    if basemap is None:
        basemap = ctx.providers.Esri.WorldStreetMap

    raster = raster_result["raster"]
    dst_crs = raster_result["dst_crs"]

    masked = np.ma.masked_where(raster <= 0, raster)
    if masked.count() == 0:
        raise ValueError("No non-zero values to display. Check inputs or parameters.")
    vmin = float(masked.min())
    vmax = float(masked.max())
    if vmin >= vmax:
        raise ValueError("Raster has constant values; adjust parameters.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    left, bottom, right, top = _get_raster_bounds(raster_result)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    try:
        ctx.add_basemap(ax, crs=dst_crs, source=basemap, zoom=zoom, reset_extent=False)
    except Exception:
        try:
            ctx.add_basemap(ax, crs=dst_crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom, reset_extent=False)
        except Exception:
            pass

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_under("white")
    use_log = vmax > 1.0 and vmin > 0

    from matplotlib.colors import LogNorm

    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax) if use_log else None

    im = ax.imshow(
        masked,
        extent=(left, right, bottom, top),
        origin="upper",
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        zorder=3,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(f"Moving window sum (range: {vmin:.0f} to {vmax:.0f})")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax


def plot_people_contours(
    raster_result,
    color="purple",
    levels=10,
    linewidths=1.2,
    linewidth_min=0.6,
    linewidth_max=3.0,
    alpha=1.0,
    show_peaks=False,
    show_peak_labels=False,
    peak_count=5,
    peak_label_count=1,
    peak_marker="x",
    peak_size=60,
    ax=None,
):
    """Plot contours from a population raster."""
    raster = raster_result["raster"]

    masked = np.ma.masked_where(raster <= 0, raster)
    if masked.count() == 0:
        raise ValueError("No non-zero values to contour. Check inputs or parameters.")
    if float(masked.min()) == float(masked.max()):
        raise ValueError("Raster has constant values; contours not meaningful.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    left, bottom, right, top = _get_raster_bounds(raster_result)
    x = np.linspace(left, right, raster.shape[1])
    y = np.linspace(top, bottom, raster.shape[0])
    xx, yy = np.meshgrid(x, y)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    if np.isscalar(levels):
        levels = np.linspace(float(masked.min()), float(masked.max()), int(levels))
    if linewidths == "scaled":
        linewidths = np.linspace(linewidth_min, linewidth_max, len(levels))
    ax.contour(xx, yy, masked, levels=levels, colors=color, linewidths=linewidths, alpha=alpha, zorder=4)

    if show_peaks or show_peak_labels:
        filled = masked.filled(-np.inf)
        flat = filled.ravel()
        peak_count = min(int(peak_count), flat.size)
        peak_idx = np.argpartition(flat, -peak_count)[-peak_count:]
        peak_idx = peak_idx[np.argsort(flat[peak_idx])[::-1]]
        rows, cols = np.unravel_index(peak_idx, filled.shape)
        if show_peaks:
            ax.scatter(
                x[cols],
                y[rows],
                s=peak_size,
                marker=peak_marker,
                color=color,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
        if show_peak_labels:
            label_count = min(int(peak_label_count), len(rows))
            for idx in range(label_count):
                value = filled[rows[idx], cols[idx]]
                ax.annotate(
                    f"{value:.0f}",
                    xy=(x[cols[idx]], y[rows[idx]]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    color=color,
                    fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
                    zorder=6,
                )
    return fig, ax


def contours_to_gdf(raster_result, levels=10, output_crs=None):
    """Convert raster contours to GeoDataFrame of LineStrings."""
    raster = raster_result["raster"]
    dst_crs = raster_result["dst_crs"]
    masked = np.ma.masked_where(raster <= 0, raster)
    if masked.count() == 0:
        raise ValueError("No non-zero values to contour.")
    if float(masked.min()) == float(masked.max()):
        raise ValueError("Raster has constant values; contours not meaningful.")

    left, bottom, right, top = _get_raster_bounds(raster_result)
    x = np.linspace(left, right, raster.shape[1])
    y = np.linspace(top, bottom, raster.shape[0])
    xx, yy = np.meshgrid(x, y)
    if np.isscalar(levels):
        levels = np.linspace(float(masked.min()), float(masked.max()), int(levels))

    fig, ax = plt.subplots(figsize=(6, 4))
    cs = ax.contour(xx, yy, masked, levels=levels)
    plt.close(fig)

    records = []
    for level, segments in zip(cs.levels, cs.allsegs):
        for segment in segments:
            if segment.shape[0] < 2:
                continue
            line = LineString(segment)
            records.append({"level": float(level), "geometry": line})
    if not records:
        raise ValueError("No contour geometries were created.")

    gdf = gpd.GeoDataFrame(records, crs=dst_crs)
    if output_crs and gdf.crs != output_crs:
        gdf = gdf.to_crs(output_crs)
    return gdf


def smooth_close_contours_gdf(gdf, simplify_tolerance=0.0001, close_tolerance=0.0001, min_area=0.0):
    """Close open contours and convert to polygons, merging overlapping contours."""
    if gdf.empty:
        return gdf.copy()

    records = []
    for level, group in gdf.groupby("level") if "level" in gdf.columns else [(None, gdf)]:
        closed_geoms = []
        for geom in group.geometry:
            if geom is None or geom.is_empty:
                continue
            geom_type = geom.geom_type
            if geom_type in ("Polygon", "MultiPolygon"):
                closed_geoms.append(geom)
                continue
            if geom_type == "LineString":
                lines = [geom]
            elif geom_type == "LinearRing":
                lines = [LineString(geom)]
            elif geom_type == "MultiLineString":
                lines = list(geom.geoms)
            else:
                continue
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 4:
                    continue
                if coords[0] != coords[-1]:
                    start = coords[0]
                    end = coords[-1]
                    dx = start[0] - end[0]
                    dy = start[1] - end[1]
                    if (dx * dx + dy * dy) ** 0.5 <= close_tolerance:
                        coords.append(start)
                    else:
                        continue
                ring = sg.LinearRing(coords)
                if not ring.is_valid:
                    continue
                closed_geoms.append(Polygon(ring))

        if not closed_geoms:
            continue

        merged = unary_union(closed_geoms)
        if merged.geom_type == "Polygon":
            polys = [merged]
        elif merged.geom_type == "MultiPolygon":
            polys = list(merged.geoms)
        else:
            polys = list(polygonize(merged))

        if simplify_tolerance and simplify_tolerance > 0:
            polys = [p.simplify(simplify_tolerance, preserve_topology=True) for p in polys]
        if min_area and min_area > 0:
            polys = [p for p in polys if p.area >= min_area]

        for poly in polys:
            if "level" in gdf.columns:
                records.append({"level": float(level), "geometry": poly})
            else:
                records.append({"geometry": poly})

    return gpd.GeoDataFrame(records, crs=gdf.crs)


def raster_to_data_url(raster_result, cmap_name="Purples", alpha=0.7):
    """Convert raster to base64 data URL for web mapping (e.g., folium)."""
    raster = raster_result["raster"]
    left, bottom, right, top = _get_raster_bounds(raster_result)
    masked = np.ma.masked_where(raster <= 0, raster)
    if masked.count() == 0:
        raise ValueError("No non-zero values to display in raster overlay.")
    vmin = float(masked.min())
    vmax = float(masked.max())
    if vmin >= vmax:
        raise ValueError("Raster has constant values; cannot build overlay.")
    if vmin > 0 and vmax > 1.0:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(masked.filled(np.nan)))
    rgba[..., 3] = np.where(masked.mask, 0.0, alpha)
    buffer = io.BytesIO()
    mpimg.imsave(buffer, rgba, format="png")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{data}"
    bounds = [[bottom, left], [top, right]]
    return data_url, bounds


def get_species_log_scale_range(air_pollution_dict, resolution, species):
    """Calculate consistent log scale range for a species across all statistics."""
    all_nonzero_values = []
    all_max_values = []

    for statistic in air_pollution_dict[resolution][species]:
        data = air_pollution_dict[resolution][species][statistic]["data"][0]
        nonzero_data = data[data > 0]
        if len(nonzero_data) > 0:
            all_nonzero_values.extend(nonzero_data.flatten())
            all_max_values.append(data.max())

    if len(all_nonzero_values) > 0:
        global_min = np.min(all_nonzero_values)
        global_max = np.max(all_max_values)
        log_vmin = global_min * 0.1
        log_vmax = global_max
    else:
        log_vmin = 1e-6
        log_vmax = 1.0

    return log_vmin, log_vmax


def plot_data_extent_with_overlay(
    air_pollution_dict,
    resolution="uifl_1km",
    species="Benzene",
    statistic="p25",
    show_data_overlay=True,
    log_scale=True,
    alpha=0.7,
    save_png=False,
    output_dir="maps",
):
    """Plot data extent polygon on basemap with optional air pollution data overlay."""
    data_dict = air_pollution_dict[resolution][species][statistic]

    bbox_geom, bounds = create_data_extent_box(data_dict)

    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=data_dict["crs"])
    bbox_wgs84 = bbox_gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(14, 12))

    boundary_alpha = 0.6 if show_data_overlay else 1.0
    fill_alpha = 0.1 if show_data_overlay else 0.2

    bbox_wgs84.boundary.plot(ax=ax, color="red", linewidth=2, alpha=boundary_alpha, label="Data Extent")
    bbox_wgs84.plot(ax=ax, alpha=fill_alpha, color="red", edgecolor="red", linewidth=2)

    try:
        ctx.add_basemap(ax, crs=bbox_wgs84.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        basemap_source = "CartoDB Positron"
    except Exception:
        try:
            ctx.add_basemap(ax, crs=bbox_wgs84.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            basemap_source = "OpenStreetMap"
        except Exception:
            print("Warning: Could not add base map")
            basemap_source = "None"

    if show_data_overlay:
        raster_data = data_dict["data"][0]
        transform = data_dict["transform"]

        if "p100" in air_pollution_dict[resolution][species]:
            scale_min = 0
            scale_max = air_pollution_dict[resolution][species]["p100"]["data"][0].max()
        else:
            scale_min = 0
            scale_max = raster_data.max()

        dst_crs = "EPSG:4326"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            data_dict["crs"], dst_crs, data_dict["width"], data_dict["height"], *data_dict["bounds"]
        )
        dst_array = np.zeros((dst_height, dst_width), dtype=raster_data.dtype)

        reproject(
            source=raster_data,
            destination=dst_array,
            src_transform=transform,
            src_crs=data_dict["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        from rasterio.bounds import array_bounds

        bnds = array_bounds(dst_height, dst_width, dst_transform)
        extent = [bnds[0], bnds[2], bnds[1], bnds[3]]

        data_mask = dst_array > 0

        if log_scale and dst_array.max() > 0:
            log_vmin, log_vmax = get_species_log_scale_range(air_pollution_dict, resolution, species)
            plot_data = np.where(dst_array <= 0, log_vmin, dst_array)
            norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
            scale_max = log_vmax
        else:
            plot_data = dst_array
            norm = plt.Normalize(vmin=scale_min, vmax=scale_max)

        masked_data = np.ma.masked_where(~data_mask, plot_data)

        im = ax.imshow(masked_data, extent=extent, alpha=alpha, cmap="viridis", norm=norm, zorder=5)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        cbar.set_label(f"{species} Concentration (ppb)", fontsize=12)

        if log_scale:
            standard_ticks = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            standard_labels = ["0.001", "0.01", "0.1", "1", "10", "100"]

            valid_ticks = []
            valid_labels = []

            for tick, label in zip(standard_ticks, standard_labels):
                if log_vmin <= tick <= scale_max:
                    valid_ticks.append(tick)
                    valid_labels.append(label)

            if valid_ticks:
                cbar.set_ticks(valid_ticks)
                cbar.set_ticklabels(valid_labels)

    cities_data = []
    for city_name, coords in TEXAS_CITIES.items():
        cities_data.append({"city": city_name, "geometry": Point(coords["lon"], coords["lat"])})

    cities_gdf = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")

    cities_gdf.plot(
        ax=ax, color="white", markersize=120, alpha=0.9, edgecolor="black", linewidth=2, zorder=15
    )
    cities_gdf.plot(
        ax=ax, color="blue", markersize=80, alpha=0.9, edgecolor="white", linewidth=1, zorder=16
    )

    for _, row in cities_gdf.iterrows():
        ax.annotate(
            row["city"],
            (row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            zorder=17,
        )

    scale_type = "with Logarithmic Scale" if (show_data_overlay and log_scale) else ""
    overlay_text = "with Data Overlay" if show_data_overlay else ""
    resolution_label = "1km" if "1km" in resolution else "4km"

    title = f"{species} {statistic} ({resolution_label})\n{overlay_text} {scale_type}"
    if show_data_overlay:
        title += f"\nData Range: {raster_data.min():.3f} - {raster_data.max():.3f} ppb"

    ax.set_title(title, fontsize=14, fontweight="bold")

    legend_elements = []
    if show_data_overlay:
        from matplotlib.patches import Patch

        legend_elements.append(Patch(facecolor="red", alpha=0.6, label="Data Extent"))

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)

    print("Enhanced Map Information:")
    print(f"  Resolution: {resolution}")
    print(f"  Species: {species}")
    print(f"  Statistic: {statistic}")
    print(f"  Data Overlay: {'Yes' if show_data_overlay else 'No'}")
    print(f"  Scale: {'Logarithmic' if (show_data_overlay and log_scale) else 'Linear'}")
    if show_data_overlay:
        print(f"  Data Range: {dst_array.min():.3f} - {dst_array.max():.3f} ppb")
        if log_scale:
            log_vmin, log_vmax = get_species_log_scale_range(air_pollution_dict, resolution, species)
            print(
                f"  Consistent Log Scale: {log_vmin:.6f} - {log_vmax:.3f} ppb (same for all {species} statistics)"
            )
        print("  Reprojected to: WGS84 (EPSG:4326)")
        print(f"  Original size: {raster_data.shape}, Reprojected size: {dst_array.shape}")
    print(f"  Base Map: {basemap_source}")

    if save_png:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_entry = air_pollution_dict[resolution][species][statistic]
        original_filename = data_entry["original_filename"]
        base_filename = original_filename.replace(".tif", "").replace(".tiff", "")
        png_filename = f"{base_filename}.png"
        save_path = os.path.join(output_dir, png_filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved map to: {save_path}")

    plt.tight_layout()
    plt.show()

    return bbox_gdf


def create_species_boxplots(air_pollution_dict, resolution="uifl_1km", figsize=(20, 15)):
    """Extract data from raster dictionary for box plot creation."""
    species_list = list(air_pollution_dict[resolution].keys())

    plot_data = []
    species_stats = {}

    print("Extracting data from rasters...")

    for species in species_list:
        species_stats[species] = {}
        stats_available = list(air_pollution_dict[resolution][species].keys())

        percentile_stats = [stat for stat in stats_available if stat.startswith("p")]
        percentile_stats = sorted(percentile_stats, key=lambda x: int(x[1:]))

        if "mean" in stats_available:
            percentile_stats.append("mean")

        for statistic in percentile_stats:
            data_array = air_pollution_dict[resolution][species][statistic]["data"][0]
            valid_data = data_array[data_array > 0]

            if len(valid_data) > 0:
                species_stats[species][statistic] = valid_data

                for value in valid_data.flatten():
                    plot_data.append(
                        {"Species": species, "Statistic": statistic, "Concentration": value}
                    )

    df = pd.DataFrame(plot_data)

    print(f"Data extracted for {len(species_list)} species")
    print(f"Total data points: {len(plot_data)}")

    return df, species_stats


def plot_species_boxplots_grid(
    df, species_stats, resolution="uifl_1km", figsize=(24, 16), programname="output", save_output=True
):
    """Create grid of box plots showing concentration distributions by species and percentile."""
    import seaborn as sns

    species_list = list(species_stats.keys())
    n_species = len(species_list)

    n_cols = 3
    n_rows = (n_species + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    plt.style.use("default")

    for idx, species in enumerate(species_list):
        ax = axes[idx]
        species_df = df[df["Species"] == species].copy()

        if len(species_df) > 0:
            sns.boxplot(
                data=species_df,
                x="Statistic",
                y="Concentration",
                ax=ax,
                hue="Statistic",
                palette="Set2",
                legend=False,
                showfliers=True,
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
            )

            ax.set_title(f"{species}\n({resolution})", fontsize=14, fontweight="bold")
            ax.set_xlabel("Percentile", fontsize=12)
            ax.set_ylabel("Concentration (ppb)", fontsize=12)

            ax.set_yscale("log")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

            stats_text = []
            for stat in sorted(species_stats[species].keys()):
                data = species_stats[species][stat]
                median_val = np.median(data)
                stats_text.append(f"{stat}: {median_val:.2f}")

            textstr = "\n".join(stats_text)
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=props)
        else:
            ax.text(0.5, 0.5, f"No data\nfor {species}", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(species, fontsize=14)

    for idx in range(n_species, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        f"Air Pollution Concentration Distributions by Species and Percentile\n{resolution} Resolution",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.92)
    plt.show()

    if save_output:
        if not os.path.exists(programname):
            os.makedirs(programname)

        filename = f"{programname}_boxplots_{resolution}.png"
        filepath = os.path.join(programname, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Box plot saved to: {filepath}")

    return fig


def create_summary_statistics_table(species_stats, resolution="uifl_1km"):
    """Create summary statistics table for all species and percentiles."""
    summary_data = []

    for species in species_stats:
        for statistic in species_stats[species]:
            data = species_stats[species][statistic]

            summary_data.append(
                {
                    "Species": species,
                    "Percentile": statistic,
                    "Count": len(data),
                    "Min": np.min(data),
                    "Q25": np.percentile(data, 25),
                    "Median": np.median(data),
                    "Q75": np.percentile(data, 75),
                    "Max": np.max(data),
                    "Mean": np.mean(data),
                    "Std": np.std(data),
                }
            )

    summary_df = pd.DataFrame(summary_data)

    print(f"\nSummary Statistics for {resolution}:")
    print("=" * 100)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(summary_df.to_string(index=False))

    return summary_df
