"""
Hotspot analysis functions for population and air pollution data.

This module provides functions for creating contour maps to identify hotspots:
- Areas with high population density
- Concentrations of vulnerable populations (low income renters, high income homeowners)
- Pollutant-weighted population exposure
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch
import folium
from folium import plugins

from ip1_0bv1_config import (
    DEFAULT_CONTOUR_PARAMS,
    DEFAULT_LOW_INCOME_PERCENTILE,
    DEFAULT_HIGH_INCOME_PERCENTILE,
    DEMOGRAPHIC_COLORS,
)
from ip1_0fv1_explore_visualize import (
    build_people_raster,
    plot_people_contours,
    contours_to_gdf,
    smooth_close_contours_gdf,
)


def filter_demographic_groups(
    gdf,
    low_income_percentile=DEFAULT_LOW_INCOME_PERCENTILE,
    high_income_percentile=DEFAULT_HIGH_INCOME_PERCENTILE,
):
    """
    Filter GeoDataFrame into demographic groups based on income and tenure.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'randincome' and 'ownershp' columns.
    low_income_percentile : float, optional
        Percentile threshold for low income (default: 0.25).
    high_income_percentile : float, optional
        Percentile threshold for high income (default: 0.75).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'low_income_renters': GeoDataFrame of low income renters
        - 'high_income_homeowners': GeoDataFrame of high income homeowners
        - 'thresholds': dict with 'low' and 'high' income values
        - 'counts': dict with counts for each group
    """
    # Calculate income thresholds
    low_threshold = gdf["randincome"].quantile(low_income_percentile)
    high_threshold = gdf["randincome"].quantile(high_income_percentile)

    # Filter demographic groups
    # ownershp: 1=owner, 2=renter
    low_income_renters = gdf[
        (gdf["ownershp"] == 2) & (gdf["randincome"] <= low_threshold)
    ].copy()
    
    high_income_homeowners = gdf[
        (gdf["ownershp"] == 1) & (gdf["randincome"] >= high_threshold)
    ].copy()

    return {
        "low_income_renters": low_income_renters,
        "high_income_homeowners": high_income_homeowners,
        "thresholds": {
            "low": low_threshold,
            "high": high_threshold,
            "low_percentile": low_income_percentile,
            "high_percentile": high_income_percentile,
        },
        "counts": {
            "low_income_renters": len(low_income_renters),
            "high_income_homeowners": len(high_income_homeowners),
            "total": len(gdf),
        },
    }


def analyze_contour_population(
    contours_gdf,
    points_gdf,
    output_dir=None,
    output_filename=None,
):
    """
    Perform spatial join between contour polygons and point data to calculate
    summary statistics for population within each contour level.

    Parameters
    ----------
    contours_gdf : GeoDataFrame
        Contour polygons with 'level' column indicating contour values.
    points_gdf : GeoDataFrame
        Point data with population and demographic attributes.
    output_dir : str, optional
        Directory to save output CSV file (default: None, no file saved).
    output_filename : str, optional
        Name for output CSV file (default: None, auto-generated).

    Returns
    -------
    DataFrame
        Summary statistics by contour level including:
        - household_count: number of households
        - total_population: sum of numprec
        - mean_income: average household income
        - median_income: median household income
        - pct_renters: percentage of renters
        - pct_homeowners: percentage of homeowners
    """
    print(f"Performing spatial join between {len(contours_gdf)} contours and {len(points_gdf)} points...")
    
    # Ensure both are in same CRS
    if contours_gdf.crs != points_gdf.crs:
        points_gdf = points_gdf.to_crs(contours_gdf.crs)
    
    # Spatial join: find which contour each point falls within
    joined = gpd.sjoin(points_gdf, contours_gdf[['level', 'geometry']], how='inner', predicate='within')
    
    print(f"  {len(joined)} points matched to contours (out of {len(points_gdf)} total)")
    
    # Calculate summary statistics by contour level
    summary = joined.groupby('level').agg({
        'numprec': ['count', 'sum', 'mean'],
        'randincome': ['mean', 'median', 'std'],
        'ownershp': lambda x: (x == 2).sum(),  # Count renters (ownershp=2)
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'numprec_count': 'household_count',
        'numprec_sum': 'total_population',
        'numprec_mean': 'mean_household_size',
        'randincome_mean': 'mean_income',
        'randincome_median': 'median_income',
        'randincome_std': 'std_income',
        'ownershp_<lambda>': 'renter_count',
    })
    
    # Add calculated percentages
    summary['pct_renters'] = (summary['renter_count'] / summary['household_count'] * 100).round(1)
    summary['pct_homeowners'] = (100 - summary['pct_renters']).round(1)
    
    # Reset index to make level a column
    summary = summary.reset_index()
    
    # Sort by contour level
    summary = summary.sort_values('level')
    
    print(f"\nSummary Statistics by Contour Level:")
    print(f"  Contour levels analyzed: {len(summary)}")
    print(f"  Total households in contours: {summary['household_count'].sum():,.0f}")
    print(f"  Total population in contours: {summary['total_population'].sum():,.0f}")
    
    # Save to CSV if requested
    if output_dir is not None and output_filename is not None:
        csv_path = os.path.join(output_dir, output_filename)
        summary.to_csv(csv_path, index=False)
        print(f"  Saved summary to: {csv_path}")
    
    return summary


def create_population_contours(
    gdf,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    contour_increment=None,
    levels=None,
    save_shapefile=True,
):
    """
    Create contour map showing total population density.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'numprec' (number of people) and geometry columns.
    output_dir : str
        Directory to save output files.
    programname : str
        Program name for output file naming.
    cell_size_m : int, optional
        Raster cell size in meters (default: 400).
    window_radius_m : int, optional
        Moving window radius in meters (default: 500).
    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for every 100 people).
        If provided, levels are calculated from raster data range.
        If not provided, uses 'levels' parameter (default: None).
    levels : int or list, optional
        Number of contour levels (int) or specific level values (list/array).
        Default: 12 if contour_increment is None.
    save_shapefile : bool, optional
        Whether to save contour shapefile (default: True).

    Returns
    -------
    dict
        Dictionary with 'raster_result' and 'contours_gdf' (if shapefile saved).
    """
    print(f"Building population density raster (cell={cell_size_m}m, window={window_radius_m}m)...")
    raster_result = build_people_raster(
        gdf, 
        cell_size_m=cell_size_m, 
        window_radius_m=window_radius_m, 
        value_col="numprec"
    )

    # Calculate contour levels if using increment
    if contour_increment is not None:
        raster_data = raster_result["raster"]
        max_val = raster_data.max()
        # Create levels from increment up to max value
        levels = list(range(int(contour_increment), int(max_val) + int(contour_increment), int(contour_increment)))
        print(f"Calculated {len(levels)} contour levels (increment={contour_increment}): {levels[0]} to {levels[-1]}")
    elif levels is None:
        levels = DEFAULT_CONTOUR_PARAMS["levels"]
        print(f"Plotting {levels} contour levels...")
    else:
        print(f"Plotting {len(levels) if isinstance(levels, (list, tuple)) else levels} contour levels...")

    # Create figure with contours
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Add basemap first
    ctx.add_basemap(ax, crs=raster_result["gdf"].crs, source=ctx.providers.CartoDB.Positron, zoom='auto', reset_extent=False, attribution=False)
    
    plot_people_contours(
        raster_result, 
        color="blue", 
        levels=levels,
        ax=ax,
        linewidths=1.5,
        alpha=0.7,
    )
    
    plt.title(f"Total Population Density - Contour Map", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Save figure
    output_path = os.path.join(output_dir, f"{programname}_total_population_contours.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()

    result = {"raster_result": raster_result}

    # Save shapefile if requested
    if save_shapefile:
        print("Converting contours to shapefile...")
        target_crs = raster_result["gdf"].crs
        # For shapefile conversion, use levels if available, otherwise default
        convert_levels = levels if isinstance(levels, (list, tuple)) else DEFAULT_CONTOUR_PARAMS["levels"]
        contours_gdf = contours_to_gdf(raster_result, levels=convert_levels, output_crs=target_crs)
        contours_gdf = smooth_close_contours_gdf(contours_gdf)
        
        shapefile_path = os.path.join(output_dir, "total_population_contours.shp")
        contours_gdf.to_file(shapefile_path)
        print(f"Saved shapefile: {shapefile_path}")
        result["contours_gdf"] = contours_gdf

    return result


def create_demographic_contours(
    gdf,
    demographic_groups,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    contour_increment=None,
    levels=None,
    save_shapefile=True,
    save_figure=True,
):
    """
    Create combined contour map showing high income homeowners and low income renters.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data (for CRS reference).
    demographic_groups : dict
        Output from filter_demographic_groups().
    output_dir : str
        Directory to save output files.
    programname : str
        Program name for output file naming.
    cell_size_m : int, optional
        Raster cell size in meters (default: 400).
    window_radius_m : int, optional
        Moving window radius in meters (default: 500).
    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for every 100 people).
        If provided, levels are calculated from raster data range.
        If not provided, uses 'levels' parameter (default: None).
    levels : int or list, optional
        Number of contour levels (int) or specific level values (list/array).
        Default: 12 if contour_increment is None.
    save_shapefile : bool, optional
        Whether to save contour shapefiles (default: True).
    save_figure : bool, optional
        Whether to create and save the matplotlib figure (default: True).

    Returns
    -------
    dict
        Dictionary with raster results and contours for both groups.
    """
    low_gdf = demographic_groups["low_income_renters"]
    high_gdf = demographic_groups["high_income_homeowners"]
    thresholds = demographic_groups["thresholds"]
    
    print(f"Building rasters for demographic groups...")
    print(f"  Low income renters: {len(low_gdf)} households (income <= ${thresholds['low']:,.0f})")
    print(f"  High income homeowners: {len(high_gdf)} households (income >= ${thresholds['high']:,.0f})")
    
    # Build rasters
    low_raster = build_people_raster(
        low_gdf, 
        cell_size_m=cell_size_m, 
        window_radius_m=window_radius_m, 
        value_col="numprec"
    )
    
    high_raster = build_people_raster(
        high_gdf, 
        cell_size_m=cell_size_m, 
        window_radius_m=window_radius_m, 
        value_col="numprec"
    )

    # Calculate contour levels if using increment
    if contour_increment is not None:
        low_raster_data = low_raster["raster"]
        high_raster_data = high_raster["raster"]
        max_val = max(low_raster_data.max(), high_raster_data.max())
        # Create levels from increment up to max value
        levels = list(range(int(contour_increment), int(max_val) + int(contour_increment), int(contour_increment)))
        print(f"Calculated {len(levels)} contour levels (increment={contour_increment}): {levels[0]} to {levels[-1]}")
    elif levels is None:
        levels = DEFAULT_CONTOUR_PARAMS["levels"]
        print(f"Plotting {levels} contour levels...")
    else:
        print(f"Plotting {len(levels) if isinstance(levels, (list, tuple)) else levels} contour levels...")

    if save_figure:
        # Create combined figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Add basemap first
        ctx.add_basemap(ax, crs=low_raster["gdf"].crs, source=ctx.providers.CartoDB.Positron, zoom='auto', reset_extent=False, attribution=False)
        
        # Plot high income homeowners in red
        plot_people_contours(
            high_raster, 
            color=DEMOGRAPHIC_COLORS["high_income_homeowners"], 
            levels=levels,
            ax=ax,
            linewidths=1.5,
            alpha=0.7,
        )
        
        # Plot low income renters in blue
        plot_people_contours(
            low_raster, 
            color=DEMOGRAPHIC_COLORS["low_income_renters"], 
            levels=levels,
            ax=ax,
            linewidths=1.5,
            alpha=0.7,
        )
        plt.title(f"Demographic Distribution - Combined Contour Map", fontsize=14, fontweight="bold")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Create custom legend patches
        legend_elements = [
            Patch(facecolor=DEMOGRAPHIC_COLORS["high_income_homeowners"], edgecolor="black", label=f"High Income (≥{int(thresholds['high_percentile']*100)}th %ile) Homeowners"),
            Patch(facecolor=DEMOGRAPHIC_COLORS["low_income_renters"], edgecolor="black", label=f"Low Income (≤{int(thresholds['low_percentile']*100)}th %ile) Renters"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
        
        # Save figure
        output_path = os.path.join(output_dir, f"{programname}_demographics_contours.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
        plt.show()

    result = {
        "low_income_raster": low_raster,
        "high_income_raster": high_raster,
    }

    # Save shapefiles if requested
    if save_shapefile:
        print("Converting contours to shapefiles...")
        target_crs = low_raster["gdf"].crs
        # For shapefile conversion, use levels if available, otherwise default
        convert_levels = levels if isinstance(levels, (list, tuple)) else DEFAULT_CONTOUR_PARAMS["levels"]
        
        # Low income renters
        low_contours = contours_to_gdf(low_raster, levels=convert_levels, output_crs=target_crs)
        low_contours = smooth_close_contours_gdf(low_contours)
        low_shapefile = os.path.join(output_dir, "low_income_renters_contours.shp")
        low_contours.to_file(low_shapefile)
        print(f"Saved shapefile: {low_shapefile}")
        result["low_income_contours"] = low_contours
        
        # High income homeowners
        high_contours = contours_to_gdf(high_raster, levels=convert_levels, output_crs=target_crs)
        high_contours = smooth_close_contours_gdf(high_contours)
        high_shapefile = os.path.join(output_dir, "high_income_homeowners_contours.shp")
        high_contours.to_file(high_shapefile)
        print(f"Saved shapefile: {high_shapefile}")
        result["high_income_contours"] = high_contours

    return result


def create_weighted_population_contours(
    gdf,
    weight_column,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    contour_increment=None,
    levels=None,
    demographic_groups=None,
    save_shapefile=True,
    save_figure=True,
    create_folium_map=False,
):
    """
    Create contour map using pre-generated weighted population column.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'numprec' and weight column.
    weight_column : str
        Name of pre-generated weight column (e.g., 'wnpbmean', 'wnpbmeanz3', 'wnpbmeanp75').
        This column should already exist in gdf (created by generate_all_weights_for_pollutant).
    output_dir : str
        Directory to save output files.
    programname : str
        Program name for output file naming.
    cell_size_m : int, optional
        Raster cell size in meters (default: 400).
    window_radius_m : int, optional
        Moving window radius in meters (default: 500).
    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for every 100 people).
        If provided, levels are calculated from raster data range.
        If not provided, uses 'levels' parameter (default: None).
    levels : int or list, optional
        Number of contour levels (int) or specific level values (list/array).
        Default: 12 if contour_increment is None.
    demographic_groups : dict, optional
        Output from filter_demographic_groups(). If provided, creates separate
        weighted contours for each demographic group (default: None).
    save_shapefile : bool, optional
        Whether to save contour shapefile (default: True).
    save_figure : bool, optional
        Whether to create and save the matplotlib figure (default: True).
    create_folium_map : bool, optional
        Whether to create an interactive folium map (default: False).
        Only works when demographic_groups is provided.

    Returns
    -------
    dict
        Dictionary with 'raster_result' and 'contours_gdf' (if shapefile saved).
        If demographic_groups provided, includes separate results for each group.
    """
    # Verify weight column exists
    if weight_column not in gdf.columns:
        raise ValueError(f"Weight column '{weight_column}' not found in GeoDataFrame. "
                        f"Available columns: {', '.join(gdf.columns)}")
    
    # Extract weight method info from column name for display
    # Example: "wnpbmean" -> "Direct", "wnpbmeanz3" -> "Z-score (3 SD)", "wnpbmeanp75" -> "Percentile (75th)"
    if "z" in weight_column:
        # Extract z value
        z_val = weight_column.split("z")[-1]
        weight_method_desc = f"Z-score ({z_val} SD)"
    elif "p" in weight_column:
        # Extract percentile value
        p_val = weight_column.split("p")[-1]
        weight_method_desc = f"Percentile ({p_val}th)"
    else:
        weight_method_desc = "Direct"
    
    # If demographic_groups provided, create separate weighted contours for each group
    if demographic_groups is not None:
        low_gdf = demographic_groups["low_income_renters"]
        high_gdf = demographic_groups["high_income_homeowners"]
        thresholds = demographic_groups["thresholds"]
        
        print(f"Building {weight_column} weighted contours for demographic groups ({weight_method_desc})...")
        print(f"  Low income renters: {len(low_gdf)} households")
        print(f"  High income homeowners: {len(high_gdf)} households")
        
        # Filter to demographic groups and use their weight column values
        low_gdf_work = low_gdf.copy()
        if weight_column not in low_gdf_work.columns:
            # Map the weight column from main gdf by index
            low_gdf_work[weight_column] = gdf.loc[low_gdf_work.index, weight_column]
        
        high_gdf_work = high_gdf.copy()
        if weight_column not in high_gdf_work.columns:
            # Map the weight column from main gdf by index
            high_gdf_work[weight_column] = gdf.loc[high_gdf_work.index, weight_column]
        
        # Build rasters using the weight column directly
        low_raster = build_people_raster(
            low_gdf_work, 
            cell_size_m=cell_size_m, 
            window_radius_m=window_radius_m, 
            value_col=weight_column
        )
        
        high_raster = build_people_raster(
            high_gdf_work, 
            cell_size_m=cell_size_m, 
            window_radius_m=window_radius_m, 
            value_col=weight_column
        )

    # Calculate contour levels
    if contour_increment is not None:
        low_raster_data = low_raster["raster"]
        high_raster_data = high_raster["raster"]
        max_val = max(low_raster_data.max(), high_raster_data.max())

        # Auto-scale: treat contour_increment as a relative percentile floor.
        # Use the fraction (contour_increment / max_val) to derive a percentile,
        # then compute levels as equally spaced from that floor to max.
        if max_val >= contour_increment:
            # Original absolute behaviour: levels at 100, 200, 300, ...
            levels = list(
                np.arange(contour_increment, max_val + contour_increment, contour_increment)
            )
        else:
            # Scale: preserve the same *proportion* of the distribution as hotspots.
            # Find the 75th percentile of non-zero raster values as the floor.
            combined = np.concatenate([
                low_raster_data[low_raster_data > 0].flatten(),
                high_raster_data[high_raster_data > 0].flatten(),
            ])
            if len(combined) == 0:
                levels = DEFAULT_CONTOUR_PARAMS["levels"]
            else:
                min_level = np.percentile(combined, 75)
                effective_increment = (max_val - min_level) / 5
                if effective_increment <= 0:
                    effective_increment = max_val / 5
                    min_level = 0
                levels = list(
                    np.arange(min_level + effective_increment,
                                max_val + effective_increment,
                                effective_increment)
                )
                print(
                    f"INFO: max raster value ({max_val:.4f}) < contour_increment ({contour_increment}). "
                    f"Auto-scaling to 75th-percentile floor={min_level:.4f}, "
                    f"increment={effective_increment:.4f} → {len(levels)} levels."
                )
        print(f"Calculated {len(levels)} contour levels: {levels[0]:.4f} to {levels[-1]:.4f}")
    elif levels is None:
        levels = DEFAULT_CONTOUR_PARAMS["levels"]
        print(f"Plotting {levels} contour levels...")
    else:
        print(f"Plotting {len(levels) if isinstance(levels, (list, tuple)) else levels} contour levels...")
        
    if save_figure:
        # Create combined figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Add basemap first
        ctx.add_basemap(ax, crs=low_raster["gdf"].crs, source=ctx.providers.CartoDB.Positron, zoom='auto', reset_extent=False, attribution=False)
        
        # Plot high income homeowners in red
        plot_people_contours(
            high_raster, 
            color=DEMOGRAPHIC_COLORS["high_income_homeowners"], 
            levels=levels,
            ax=ax,
            linewidths=1.5,
            alpha=0.7,
        )
        
        # Plot low income renters in blue
        plot_people_contours(
            low_raster, 
            color=DEMOGRAPHIC_COLORS["low_income_renters"], 
            levels=levels,
            ax=ax,
            linewidths=1.5,
            alpha=0.7,
        )
        plt.title(
            f"{weight_column} Weighted Population Exposure by Demographics\n({weight_method_desc})", 
            fontsize=14, 
            fontweight="bold"
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Create custom legend patches
        legend_elements = [
            Patch(facecolor=DEMOGRAPHIC_COLORS["high_income_homeowners"], edgecolor="black", 
                  label=f"High Income (≥{int(thresholds['high_percentile']*100)}th %ile) Homeowners"),
            Patch(facecolor=DEMOGRAPHIC_COLORS["low_income_renters"], edgecolor="black", 
                  label=f"Low Income (≤{int(thresholds['low_percentile']*100)}th %ile) Renters"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
        
        # Save figure
        output_path = os.path.join(output_dir, f"{programname}_weighted_demographics_{weight_column}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
        plt.show()

    result = {
        "low_income_raster": low_raster,
        "high_income_raster": high_raster,
    }

    # Save shapefiles if requested
    if save_shapefile:
        print("Converting contours to shapefiles...")
        target_crs = low_raster["gdf"].crs
        convert_levels = levels if isinstance(levels, (list, tuple)) else DEFAULT_CONTOUR_PARAMS["levels"]
        
        # Low income renters
        low_contours = contours_to_gdf(low_raster, levels=convert_levels, output_crs=target_crs)
        low_contours = smooth_close_contours_gdf(low_contours)
        low_shapefile = os.path.join(output_dir, f"{weight_column}_low_income_renters.shp")
        low_contours.to_file(low_shapefile)
        print(f"Saved shapefile: {low_shapefile}")
        result["low_income_contours"] = low_contours
        
        # High income homeowners
        high_contours = contours_to_gdf(high_raster, levels=convert_levels, output_crs=target_crs)
        high_contours = smooth_close_contours_gdf(high_contours)
        high_shapefile = os.path.join(output_dir, f"{weight_column}_high_income_homeowners.shp")
        high_contours.to_file(high_shapefile)
        print(f"Saved shapefile: {high_shapefile}")
        result["high_income_contours"] = high_contours

    # Create folium map if requested
    if create_folium_map:
        print("Creating interactive folium map...")
        
        # Convert contours to WGS84 for folium
        low_contours_wgs84 = result["low_income_contours"].to_crs("EPSG:4326")
        high_contours_wgs84 = result["high_income_contours"].to_crs("EPSG:4326")
        
        # Calculate center point
        bounds = low_contours_wgs84.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='CartoDB positron'
        )
        
        # Add high income homeowners contours (red)
        folium.GeoJson(
            high_contours_wgs84,
            name=f"High Income (≥{int(thresholds['high_percentile']*100)}th %ile) Homeowners",
            style_function=lambda x: {
                'fillColor': DEMOGRAPHIC_COLORS["high_income_homeowners"],
                'color': 'darkred',
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['level'],
                aliases=['Weighted Population:'],
                labels=True
            )
        ).add_to(m)
        
        # Add low income renters contours (blue)
        folium.GeoJson(
            low_contours_wgs84,
            name=f"Low Income (≤{int(thresholds['low_percentile']*100)}th %ile) Renters",
            style_function=lambda x: {
                'fillColor': DEMOGRAPHIC_COLORS["low_income_renters"],
                'color': 'darkblue',
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['level'],
                aliases=['Weighted Population:'],
                labels=True
            )
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title
        title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50%; transform: translateX(-50%);
                        width: auto; height: auto;
                        background-color: white; border:2px solid grey;
                        z-index:9999; font-size:16px; font-weight:bold;
                        padding: 10px; border-radius: 5px;">
                {weight_column} Weighted Exposure ({weight_method_desc})
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save folium map
        folium_path = os.path.join(output_dir, f"{programname}_{weight_column}_interactive.html")
        m.save(folium_path)
        print(f"Saved interactive map: {folium_path}")
        result["folium_map_path"] = folium_path

    # Return result for demographic groups
    return result

# ============================================================================

def extract_hap_and_stat(pollutant_column):
    """
    Extract HAP abbreviation and statistic from pollutant column name.

    Parameters
    ----------
    pollutant_column : str
        Column name like "uifl_1km_Benzene_mean" or "uifl_1km_Toluene_p95"

    Returns
    -------
    tuple
        (hap_abbreviation, statistic) e.g., ("B", "mean") or ("T", "p95")
    """
    from ip1_0bv1_config import HAP_ABBREVIATIONS
    
    # Split on underscore and extract parts
    parts = pollutant_column.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid pollutant column format: {pollutant_column}")
    
    # Extract HAP name (3rd part) and statistic (4th part)
    hap_name = parts[2]  # e.g., "Benzene"
    stat = parts[3]  # e.g., "mean", "p95"
    
    # Get abbreviation
    hap_abbrev = HAP_ABBREVIATIONS.get(hap_name, hap_name[0])  # Default to first letter if not found
    
    return hap_abbrev, stat


def build_weight_column_name(hap_abbrev, stat, method, threshold=None):
    """
    Build standardized weight column name.

    Parameters
    ----------
    hap_abbrev : str
        HAP abbreviation (e.g., "B" for Benzene)
    stat : str
        Statistic (e.g., "mean", "p95")
    method : str
        Weight method: "direct", "zscore", "percentile"
    threshold : int or float, optional
        Threshold value (required for zscore and percentile)

    Returns
    -------
    str
        Column name like "wnpbmean", "wnpbmeanz3", "wnpbmeanp75"
    """
    from ip1_0bv1_config import WEIGHT_METHOD_SUFFIXES
    
    base_name = f"wnp{hap_abbrev}{stat}"
    
    if method == "direct":
        return base_name
    elif method == "zscore":
        if threshold is None:
            raise ValueError("threshold required for zscore method")
        suffix = WEIGHT_METHOD_SUFFIXES["zscore"](threshold)
        return f"{base_name}{suffix}"
    elif method == "percentile":
        if threshold is None:
            raise ValueError("threshold required for percentile method")
        suffix = WEIGHT_METHOD_SUFFIXES["percentile"](threshold)
        return f"{base_name}{suffix}"
    else:
        raise ValueError(f"Unknown weight method: {method}")


def get_pollutant_columns(gdf):
    """
    Get list of all pollutant columns from GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data

    Returns
    -------
    list
        Column names matching pattern "uifl_1km_*"
    """
    return [col for col in gdf.columns if col.startswith("uifl_1km_")]


def merge_contours_to_hotspots(contour_sets, popair_gdf, pollutant):
    """
    Merge overlapping contour polygons by method and calculate statistics.

    For each entry in contour_sets, all contour polygons are dissolved into
    unified geometry(ies) using unary_union. Statistics are calculated from
    the original point data that falls inside each merged polygon.

    Parameters
    ----------
    contour_sets : dict
        Dictionary mapping method name to (contours_gdf, weight_col) tuples.
        e.g. {'direct': (gdf, 'wnpBmean'), 'zscore': (gdf, 'wnpBmeanz3'), ...}
    popair_gdf : GeoDataFrame
        Original point data with population and pollutant columns.
    pollutant : str
        Pollutant column name (e.g. "uifl_1km_Benzene_mean").

    Returns
    -------
    GeoDataFrame
        One row per merged polygon per method, with demographic, income,
        pollutant, and weight statistics. Returns an empty GeoDataFrame if
        no merged polygons contain any points.
    """
    from shapely.ops import unary_union

    merged_hotspots_list = []

    for method_name, (contours, weight_col) in contour_sets.items():
        print(f"\n{method_name.upper()} METHOD ({weight_col}):")
        print("-" * 70)

        # Reproject contours to match popair_gdf CRS if needed
        if contours.crs != popair_gdf.crs:
            print(f"  Reprojecting from {contours.crs} to {popair_gdf.crs}...")
            contours_reprojected = contours.to_crs(popair_gdf.crs)
        else:
            contours_reprojected = contours

        # Dissolve all overlapping polygons within this method into unified geometry
        merged_geometry = unary_union(contours_reprojected.geometry)

        if merged_geometry.geom_type == 'MultiPolygon':
            merged_polygons = list(merged_geometry.geoms)
        elif merged_geometry.geom_type == 'Polygon':
            merged_polygons = [merged_geometry]
        else:
            print(f"  Warning: Unexpected geometry type: {merged_geometry.geom_type}")
            continue

        print(f"  Merged into {len(merged_polygons)} unified polygon(s)")

        for poly_idx, merged_poly in enumerate(merged_polygons, 1):
            merged_id = f"{weight_col}_merged_{poly_idx:03d}"

            temp_gdf = gpd.GeoDataFrame(
                {'temp_id': [0], 'geometry': [merged_poly]}, crs=popair_gdf.crs
            )
            points_joined = gpd.sjoin(popair_gdf, temp_gdf, how='inner', predicate='intersects')

            if len(points_joined) == 0:
                print(f"    Polygon {poly_idx}: No points found inside, skipping")
                continue

            point_indices = points_joined.index.unique()
            points_in_merged = popair_gdf.loc[point_indices]

            # Demographic statistics
            total_population = points_in_merged['numprec'].sum()

            income_values = points_in_merged['randincome'].dropna()
            median_income = income_values.median() if len(income_values) > 0 else np.nan

            poverty_values = points_in_merged['poverty'].dropna()
            total_with_poverty_values = len(poverty_values)
            poverty_rate = (
                poverty_values.sum() / total_with_poverty_values * 100
                if total_with_poverty_values > 0 else np.nan
            )

            total_population_renters = points_in_merged.loc[
                points_in_merged['ownershp'] == 2, 'numprec'
            ].sum()
            renter_pct = (total_population_renters / total_population * 100) if total_population > 0 else 0
            total_population_owners = points_in_merged.loc[
                points_in_merged['ownershp'] == 1, 'numprec'
            ].sum()
            owner_pct = (total_population_owners / total_population * 100) if total_population > 0 else 0

            # Pollutant statistics
            hap_values = points_in_merged[pollutant].dropna()
            hap_mean = hap_values.mean() if len(hap_values) > 0 else np.nan
            hap_min = hap_values.min() if len(hap_values) > 0 else np.nan
            hap_max = hap_values.max() if len(hap_values) > 0 else np.nan

            # Weight statistics
            weight_values = points_in_merged[weight_col].dropna()
            weight_total = weight_values.sum()
            weight_mean = weight_values.mean() if len(weight_values) > 0 else np.nan
            weight_max = weight_values.max() if len(weight_values) > 0 else np.nan

            area_m2 = gpd.GeoSeries([merged_poly], crs=popair_gdf.crs).to_crs('EPSG:32615').area.iloc[0]
            area_km2 = area_m2 / 1_000_000

            merged_hotspots_list.append({
                'hotspot_id': merged_id,
                'method': method_name,
                'weight_column': weight_col,
                'hap_pollutant': pollutant,
                'merged_polygon_num': poly_idx,
                'geometry': merged_poly,
                'area_km2': area_km2,
                'area_m2': area_m2,
                'total_population': total_population,
                'total_population_renters': total_population_renters,
                'total_population_owners': total_population_owners,
                'renter_pct': renter_pct,
                'owner_pct': owner_pct,
                'median_income': median_income,
                'poverty_rate': poverty_rate,
                'hap_mean_conc': hap_mean,
                'hap_min_conc': hap_min,
                'hap_max_conc': hap_max,
                'weighted_pop_total': weight_total,
                'weighted_pop_mean': weight_mean,
                'weighted_pop_max': weight_max,
            })

            print(f"    Polygon {poly_idx}: {total_population:,.0f} people, area={area_km2:.4f} km²")

    if len(merged_hotspots_list) > 0:
        merged_df = pd.DataFrame(merged_hotspots_list)
        return gpd.GeoDataFrame(merged_df, geometry='geometry', crs=popair_gdf.crs)
    else:
        print("\nWarning: No merged hotspots found with points inside! Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(
            columns=['hotspot_id', 'method', 'weight_column', 'hap_pollutant', 'geometry'],
            crs=popair_gdf.crs,
        )


# ============================================================================
# Weight calculation functions
# ============================================================================

def apply_direct_weight(gdf, pollutant_column, inplace=False):
    """
    Apply direct multiplication weighting: weight = population × pollutant.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'numprec' and pollutant column
    pollutant_column : str
        Name of pollutant column (e.g., "uifl_1km_Benzene_mean")
    inplace : bool, optional
        If True, modify gdf. Otherwise return new GeoDataFrame (default: False)

    Returns
    -------
    GeoDataFrame
        Data with new weight column added
    """
    gdf_work = gdf if inplace else gdf.copy()
    
    hap_abbrev, stat = extract_hap_and_stat(pollutant_column)
    col_name = build_weight_column_name(hap_abbrev, stat, "direct")
    
    # Direct multiplication
    gdf_work[col_name] = gdf_work["numprec"] * gdf_work[pollutant_column]
    
    print(f"Created: {col_name}")
    print(f"  Range: {gdf_work[col_name].min():.4f} to {gdf_work[col_name].max():.4f}")
    print(f"  Mean: {gdf_work[col_name].mean():.4f}")
    
    return gdf_work


def apply_zscore_weight(gdf, pollutant_column, threshold_sd=3, inplace=False):
    """
    Apply z-score weighting with threshold: weight = population × max(0, zscore - threshold).

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'numprec' and pollutant column
    pollutant_column : str
        Name of pollutant column (e.g., "uifl_1km_Benzene_mean")
    threshold_sd : float, optional
        Standard deviations above mean to threshold (default: 3)
        Values below this threshold are zeroed out
    inplace : bool, optional
        If True, modify gdf. Otherwise return new GeoDataFrame (default: False)

    Returns
    -------
    GeoDataFrame
        Data with new weight column added
    """
    gdf_work = gdf if inplace else gdf.copy()
    
    hap_abbrev, stat = extract_hap_and_stat(pollutant_column)
    col_name = build_weight_column_name(hap_abbrev, stat, "zscore", threshold_sd)
    
    # Calculate z-scores
    pollutant_data = gdf_work[pollutant_column]
    mean_val = pollutant_data.mean()
    std_val = pollutant_data.std()
    zscore = (pollutant_data - mean_val) / std_val
    
    # Apply threshold: only values above threshold_sd contribute
    threshold_limit = zscore - threshold_sd
    weight_multiplier = np.maximum(threshold_limit, 0)  # 0 below threshold
    
    gdf_work[col_name] = gdf_work["numprec"] * weight_multiplier
    
    # Calculate statistics for reporting
    non_zero_count = (gdf_work[col_name] > 0).sum()
    
    print(f"Created: {col_name} (threshold: {threshold_sd} SD above mean)")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Threshold value: {mean_val + threshold_sd * std_val:.4f}")
    print(f"  Non-zero cells: {non_zero_count} / {len(gdf_work)} ({non_zero_count/len(gdf_work)*100:.1f}%)")
    print(f"  Range: {gdf_work[col_name].min():.4f} to {gdf_work[col_name].max():.4f}")
    print(f"  Mean: {gdf_work[col_name].mean():.4f}")
    
    return gdf_work


def apply_percentile_weight(gdf, pollutant_column, threshold_pct=75, inplace=False):
    """
    Apply percentile-based weighting: weight = population × normalized_above_threshold.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'numprec' and pollutant column
    pollutant_column : str
        Name of pollutant column (e.g., "uifl_1km_Benzene_mean")
    threshold_pct : float, optional
        Percentile threshold (default: 75)
        Values below this percentile are zeroed out
        Values above are scaled 0-1 relative to range above threshold
    inplace : bool, optional
        If True, modify gdf. Otherwise return new GeoDataFrame (default: False)

    Returns
    -------
    GeoDataFrame
        Data with new weight column added
    """
    gdf_work = gdf if inplace else gdf.copy()
    
    hap_abbrev, stat = extract_hap_and_stat(pollutant_column)
    col_name = build_weight_column_name(hap_abbrev, stat, "percentile", threshold_pct)
    
    # Calculate percentile threshold
    pollutant_data = gdf_work[pollutant_column]
    threshold_val = pollutant_data.quantile(threshold_pct / 100.0)
    max_val = pollutant_data.max()
    
    # Apply threshold: scale values above threshold from 0 to 1
    above_threshold = (pollutant_data - threshold_val) / (max_val - threshold_val)
    weight_multiplier = np.maximum(above_threshold, 0)  # 0 below threshold
    
    gdf_work[col_name] = gdf_work["numprec"] * weight_multiplier
    
    # Calculate statistics for reporting
    non_zero_count = (gdf_work[col_name] > 0).sum()
    
    print(f"Created: {col_name} (threshold: {threshold_pct}th percentile)")
    print(f"  Threshold value: {threshold_val:.4f}")
    print(f"  Max value: {max_val:.4f}")
    print(f"  Non-zero cells: {non_zero_count} / {len(gdf_work)} ({non_zero_count/len(gdf_work)*100:.1f}%)")
    print(f"  Range: {gdf_work[col_name].min():.4f} to {gdf_work[col_name].max():.4f}")
    print(f"  Mean: {gdf_work[col_name].mean():.4f}")
    
    return gdf_work


def generate_all_weights_for_pollutant(
    gdf,
    pollutant_column,
    methods=None,
    zscore_thresholds=None,
    percentile_thresholds=None,
    inplace=False,
):
    """
    Generate all weight columns for a single pollutant using specified methods.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with 'numprec' and pollutant columns
    pollutant_column : str
        Name of pollutant column (e.g., "uifl_1km_Benzene_mean")
    methods : list, optional
        Weight methods to apply (default: ['direct', 'zscore', 'percentile'])
        Options: 'direct', 'zscore', 'percentile'
    zscore_thresholds : list, optional
        Z-score thresholds to use (default: [3])
        Example: [2, 3] creates z2 and z3 versions
    percentile_thresholds : list, optional
        Percentile thresholds to use (default: [75])
        Example: [50, 75] creates p50 and p75 versions
    inplace : bool, optional
        If True, modify gdf. Otherwise return new GeoDataFrame (default: False)

    Returns
    -------
    GeoDataFrame
        Data with all requested weight columns added
    """
    if methods is None:
        methods = ["direct", "zscore", "percentile"]
    if zscore_thresholds is None:
        zscore_thresholds = [3]
    if percentile_thresholds is None:
        percentile_thresholds = [75]
    
    gdf_work = gdf if inplace else gdf.copy()
    
    hap_abbrev, stat = extract_hap_and_stat(pollutant_column)
    print(f"\n{'='*70}")
    print(f"Generating weights for {pollutant_column} ({hap_abbrev}-{stat})")
    print(f"{'='*70}")
    
    created_cols = []
    
    # Direct weighting
    if "direct" in methods:
        print(f"\n--- Direct Weighting ---")
        gdf_work = apply_direct_weight(gdf_work, pollutant_column, inplace=True)
        col_name = build_weight_column_name(hap_abbrev, stat, "direct")
        created_cols.append(col_name)
    
    # Z-score weighting
    if "zscore" in methods:
        print(f"\n--- Z-Score Weighting ---")
        for threshold in zscore_thresholds:
            gdf_work = apply_zscore_weight(gdf_work, pollutant_column, threshold_sd=threshold, inplace=True)
            col_name = build_weight_column_name(hap_abbrev, stat, "zscore", threshold)
            created_cols.append(col_name)
    
    # Percentile weighting
    if "percentile" in methods:
        print(f"\n--- Percentile Weighting ---")
        for threshold in percentile_thresholds:
            gdf_work = apply_percentile_weight(gdf_work, pollutant_column, threshold_pct=threshold, inplace=True)
            col_name = build_weight_column_name(hap_abbrev, stat, "percentile", threshold)
            created_cols.append(col_name)
    
    print(f"\n{'='*70}")
    print(f"Summary: Created {len(created_cols)} weight column(s)")
    print(f"Columns: {', '.join(created_cols)}")
    print(f"{'='*70}\n")
    
    return gdf_work


# ============================================================================
# Interactive folium map functions for merged hotspots
# ============================================================================

def _compute_area_stats(points_gdf, pollutant, area_km2=None):
    """Compute summary statistics dict from a points GeoDataFrame.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Points with population and pollutant columns.
    pollutant : str
        Pollutant column name.
    area_km2 : float or None, optional
        Land area in km². If None, computed from convex hull of points
        projected to EPSG:32615 (UTM Zone 15N).
    """
    total_pop = points_gdf['numprec'].sum()
    renter_pop = points_gdf.loc[points_gdf['ownershp'] == 2, 'numprec'].sum()
    owner_pop = points_gdf.loc[points_gdf['ownershp'] == 1, 'numprec'].sum()
    renter_pct = (renter_pop / total_pop * 100) if total_pop > 0 else 0
    owner_pct = (owner_pop / total_pop * 100) if total_pop > 0 else 0
    income_vals = points_gdf['randincome'].dropna()
    median_income = income_vals.median() if len(income_vals) > 0 else np.nan
    income_p25 = income_vals.quantile(0.25) if len(income_vals) > 0 else np.nan
    income_p75 = income_vals.quantile(0.75) if len(income_vals) > 0 else np.nan
    hap_vals = points_gdf[pollutant].dropna()

    if area_km2 is None:
        hull_area_m2 = (
            points_gdf.to_crs('EPSG:32615').unary_union.convex_hull.area
            if len(points_gdf) >= 3 else 0
        )
        area_km2 = hull_area_m2 / 1_000_000

    pop_density = total_pop / area_km2 if area_km2 > 0 else np.nan

    return {
        'total_pop': total_pop,
        'renter_pop': renter_pop,
        'owner_pop': owner_pop,
        'renter_pct': renter_pct,
        'owner_pct': owner_pct,
        'median_income': median_income,
        'income_p25': income_p25,
        'income_p75': income_p75,
        'area_km2': area_km2,
        'pop_density': pop_density,
        'hap_median': hap_vals.median() if len(hap_vals) > 0 else np.nan,
        'hap_min': hap_vals.min() if len(hap_vals) > 0 else np.nan,
        'hap_max': hap_vals.max() if len(hap_vals) > 0 else np.nan,
        'hap_p95': hap_vals.quantile(0.95) if len(hap_vals) > 0 else np.nan,
    }


def _build_stats_legend_html(title, stats, pollutant_name, position_css):
    """Build a fixed-position HTML stats inset div for a folium map."""
    area_str = f"{stats['area_km2']:,.1f}" if not np.isnan(stats['area_km2']) else 'N/A'
    density_str = f"{stats['pop_density']:,.1f}" if not np.isnan(stats['pop_density']) else 'N/A'
    return f'''
    <div style="position: fixed; {position_css}
                width: 300px; height: auto;
                background-color: white; border: 2px solid #333;
                z-index: 9998; font-size: 11px;
                padding: 12px; border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
        <p style="margin-top: 0; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 6px;">
            <u>{title}</u>
        </p>
        <p style="margin: 4px 0;"><b>Total Land Area:</b> {area_str} km&sup2;</p>
        <p style="margin: 4px 0;"><b>Total Population:</b> {stats['total_pop']:,.0f}</p>
        <p style="margin: 4px 0;"><b>Pop. Density:</b> {density_str} /km&sup2;</p>
        <p style="margin: 4px 0;"><b>Renters:</b> {stats['renter_pop']:,.0f} ({stats['renter_pct']:.1f}%)</p>
        <p style="margin: 4px 0;"><b>Owners:</b> {stats['owner_pop']:,.0f} ({stats['owner_pct']:.1f}%)</p>
        <p style="margin: 4px 0;"><b>Median Income:</b> ${stats['median_income']:,.0f}<br/>
        &nbsp;&nbsp;P25: ${stats['income_p25']:,.0f} &bull; P75: ${stats['income_p75']:,.0f}</p>
        <p style="margin: 4px 0;"><b>{pollutant_name} Conc (ppb):</b><br/>
        &nbsp;&nbsp;Median: {stats['hap_median']:.4f}<br/>
        &nbsp;&nbsp;Min: {stats['hap_min']:.4f}<br/>
        &nbsp;&nbsp;Max: {stats['hap_max']:.4f}<br/>
        &nbsp;&nbsp;95th pct: {stats['hap_p95']:.4f}</p>
    </div>
    '''


def _build_methods_legend_html():
    """Build the demographic color-key and weighting-methods reference HTML div."""
    return '''
    <div style="position: fixed;
                bottom: 50px; right: 10px;
                width: 260px; height: auto;
                background-color: white; border: 2px solid #333;
                z-index: 9998; font-size: 12px;
                padding: 12px; border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
        <p style="margin-top: 0; font-weight: bold;"><u>Demographic Groups</u></p>
        <p style="margin: 8px 0;">
            <span style="display:inline-block; width:18px; height:18px;
                         background-color:#7B4FBF; border:2px solid #4A2580;
                         border-radius:3px; margin-right:8px;"></span>
            <b>Low Income Renters</b>
        </p>
        <p style="margin: 8px 0;">
            <span style="display:inline-block; width:18px; height:18px;
                         background-color:#FF8C00; border:2px solid #CC6200;
                         border-radius:3px; margin-right:8px;"></span>
            <b>High Income Homeowners</b>
        </p>
        <p style="margin-top: 10px; font-weight: bold;"><u>Weighting Method (see popup)</u></p>
        <p style="margin: 4px 0; font-size: 10px; color: #555;">
            Direct &bull; Z-Score (3 SD) &bull; Percentile (75th)
        </p>
    </div>
    '''


def create_merged_hotspots_map(
    merged_low_gdf,
    merged_high_gdf,
    popair_gdf,
    pollutant,
    programname,
    output_dir,
    grid_gdf=None,
    grid_percentile=0.95,
):
    """
    Create an interactive folium map of merged hotspot polygons for both
    low income renters and high income homeowners, with three stats inset panels.

    Parameters
    ----------
    merged_low_gdf : GeoDataFrame
        Merged hotspot polygons for low income renters.
    merged_high_gdf : GeoDataFrame
        Merged hotspot polygons for high income homeowners.
    popair_gdf : GeoDataFrame
        Original point data (all households) for region-wide statistics.
    pollutant : str
        Pollutant column name (e.g. "uifl_1km_Benzene_mean").
    programname : str
        Program name used for the output filename.
    output_dir : str
        Directory to save the HTML output file.
    grid_gdf : GeoDataFrame or None, optional
        1km grid polygons with pollutant values. If None, grid layer is skipped.
    grid_percentile : float, optional
        Quantile threshold above which grid cells are colored (default: 0.95).

    Returns
    -------
    folium.Map
        The constructed folium map (also saved to output_dir).
    """
    import matplotlib.colors as mcolors

    pollutant_name = pollutant.split('_')[-2].capitalize()

    # Map center from union of both GDF bounds
    combined_bounds = np.array([
        merged_low_gdf.total_bounds,
        merged_high_gdf.total_bounds,
    ])
    center_lat = (combined_bounds[:, 1].min() + combined_bounds[:, 3].max()) / 2
    center_lon = (combined_bounds[:, 0].min() + combined_bounds[:, 2].max()) / 2

    hotspots_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='CartoDB positron',
    )

    # Low income renters layer (purple)
    low_layer = folium.FeatureGroup(name='Low Income Renters Hotspots', show=True)
    for _, row in merged_low_gdf.to_crs('EPSG:4326').iterrows():
        _pop_density_low = row['total_population'] / row['area_km2'] if row['area_km2'] > 0 else float('nan')
        popup_html = f"""
        <html><head><meta charset="utf-8"/></head><body>
        <div style="font-family:Arial; font-size:11px; width:280px;">
            <h4 style="margin:0 0 8px 0; border-bottom:2px solid #4A2580;
                       padding-bottom:4px; color:#4A2580;">
                Low Income Renters &mdash; {row['hotspot_id']}
            </h4>
            <b>Method:</b> {row['method'].upper()}<br/>
            <b>Weight Col:</b> {row['weight_column']}<br/>
            <hr style="margin:6px 0; border:none; border-top:1px solid #ddd;">
            <b>Area:</b> {row['area_km2']:.3f} km&sup2;<br/>
            <b>Total Population:</b> {row['total_population']:,.0f}<br/>
            <b>Pop. Density:</b> {_pop_density_low:,.1f} /km&sup2;<br/>
            <b>Renters:</b> {row['total_population_renters']:,.0f} ({row['renter_pct']:.1f}%)<br/>
            <b>Median Income:</b> ${row['median_income']:,.0f}<br/>
            <b>Poverty Rate:</b> {row['poverty_rate']:.1f}%<br/>
            <b>{row['hap_pollutant'].split('_')[-2]} Mean:</b> {row['hap_mean_conc']:.4f} ppb<br/>
        </div></body></html>
        """
        folium.GeoJson(
            gpd.GeoSeries([row['geometry']]).__geo_interface__,
            style_function=lambda x: {
                'fillColor': '#7B4FBF',
                'color': '#4A2580',
                'weight': 2,
                'fillOpacity': 0.5,
            },
            popup=folium.Popup(folium.IFrame(html=popup_html, width=320, height=300), max_width=320),
            tooltip=row['hotspot_id'],
        ).add_to(low_layer)
    low_layer.add_to(hotspots_map)

    # High income homeowners layer (orange)
    high_layer = folium.FeatureGroup(name='High Income Homeowners Hotspots', show=True)
    for _, row in merged_high_gdf.to_crs('EPSG:4326').iterrows():
        _pop_density_high = row['total_population'] / row['area_km2'] if row['area_km2'] > 0 else float('nan')
        popup_html = f"""
        <html><head><meta charset="utf-8"/></head><body>
        <div style="font-family:Arial; font-size:11px; width:280px;">
            <h4 style="margin:0 0 8px 0; border-bottom:2px solid #CC6200;
                       padding-bottom:4px; color:#CC6200;">
                High Income Homeowners &mdash; {row['hotspot_id']}
            </h4>
            <b>Method:</b> {row['method'].upper()}<br/>
            <b>Weight Col:</b> {row['weight_column']}<br/>
            <hr style="margin:6px 0; border:none; border-top:1px solid #ddd;">
            <b>Area:</b> {row['area_km2']:.3f} km&sup2;<br/>
            <b>Total Population:</b> {row['total_population']:,.0f}<br/>
            <b>Pop. Density:</b> {_pop_density_high:,.1f} /km&sup2;<br/>
            <b>Owners:</b> {row['total_population_owners']:,.0f} ({row['owner_pct']:.1f}%)<br/>
            <b>Median Income:</b> ${row['median_income']:,.0f}<br/>
            <b>Poverty Rate:</b> {row['poverty_rate']:.1f}%<br/>
            <b>{row['hap_pollutant'].split('_')[-2]} Mean:</b> {row['hap_mean_conc']:.4f} ppb<br/>
        </div></body></html>
        """
        folium.GeoJson(
            gpd.GeoSeries([row['geometry']]).__geo_interface__,
            style_function=lambda x: {
                'fillColor': '#FF8C00',
                'color': '#CC6200',
                'weight': 2,
                'fillOpacity': 0.5,
            },
            popup=folium.Popup(folium.IFrame(html=popup_html, width=320, height=300), max_width=320),
            tooltip=row['hotspot_id'],
        ).add_to(high_layer)
    high_layer.add_to(hotspots_map)

    # Optional grid layer
    if grid_gdf is not None:
        grid_crs = merged_low_gdf.crs
        grid_plot = grid_gdf.to_crs(grid_crs) if grid_gdf.crs != grid_crs else grid_gdf
        threshold_val = grid_plot[pollutant].quantile(grid_percentile)
        vmin = threshold_val
        vmax = grid_plot[pollutant].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap('YlOrRd')

        grid_layer = folium.FeatureGroup(name=f'{pollutant_name} Grid (1km)', show=False)
        for _, grow in grid_plot.to_crs('EPSG:4326').iterrows():
            cell_value = grow[pollutant]
            if cell_value >= threshold_val:
                rgba = colormap(norm(cell_value))
                fill_color = mcolors.rgb2hex(rgba)
                fill_opacity = 0.6
            else:
                fill_color = 'none'
                fill_opacity = 0
            folium.GeoJson(
                gpd.GeoSeries([grow['geometry']]).__geo_interface__,
                style_function=lambda x, fc=fill_color, fo=fill_opacity: {
                    'fillColor': fc,
                    'color': 'black',
                    'weight': 0.1,
                    'fillOpacity': fo,
                },
                tooltip=f'{pollutant_name}: {cell_value:.4f} ppb',
            ).add_to(grid_layer)
        grid_layer.add_to(hotspots_map)

    folium.LayerControl(position='topright', collapsed=False).add_to(hotspots_map)

    # Title
    title_html = f'''
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                background-color:white; border:3px solid #333;
                z-index:9999; font-size:14px; font-weight:bold;
                padding:12px 20px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);">
        Merged Hotspots &mdash; {pollutant_name} Exposure<br/>
        <span style="font-size:11px; font-weight:normal;">
            Low Income Renters (purple) &amp; High Income Homeowners (orange)
            | Click polygons for details
        </span>
    </div>
    '''
    hotspots_map.get_root().html.add_child(folium.Element(title_html))

    # Color / methods legend
    hotspots_map.get_root().html.add_child(folium.Element(_build_methods_legend_html()))

    # Three stats inset panels
    region_stats = _compute_area_stats(popair_gdf, pollutant)
    hotspots_map.get_root().html.add_child(folium.Element(
        _build_stats_legend_html(
            'Southeast Texas (All Data)', region_stats, pollutant_name,
            'bottom: 50px; left: 10px;',
        )
    ))

    low_points = gpd.sjoin(
        popair_gdf, merged_low_gdf.to_crs(popair_gdf.crs), how='inner', predicate='intersects'
    )
    if len(low_points) > 0:
        low_area_km2 = merged_low_gdf['area_km2'].sum()
        low_stats = _compute_area_stats(
            low_points.loc[~low_points.index.duplicated()], pollutant, area_km2=low_area_km2
        )
    else:
        low_stats = {k: 0 for k in ['total_pop', 'renter_pop', 'owner_pop', 'renter_pct',
                                     'owner_pct', 'median_income', 'income_p25', 'income_p75',
                                     'area_km2', 'pop_density',
                                     'hap_median', 'hap_min', 'hap_max', 'hap_p95']}
    hotspots_map.get_root().html.add_child(folium.Element(
        _build_stats_legend_html(
            'Low Income Renters Hotspots', low_stats, pollutant_name,
            'bottom: 50px; left: 320px;',
        )
    ))

    high_points = gpd.sjoin(
        popair_gdf, merged_high_gdf.to_crs(popair_gdf.crs), how='inner', predicate='intersects'
    )
    if len(high_points) > 0:
        high_area_km2 = merged_high_gdf['area_km2'].sum()
        high_stats = _compute_area_stats(
            high_points.loc[~high_points.index.duplicated()], pollutant, area_km2=high_area_km2
        )
    else:
        high_stats = {k: 0 for k in ['total_pop', 'renter_pop', 'owner_pop', 'renter_pct',
                                      'owner_pct', 'median_income', 'income_p25', 'income_p75',
                                      'area_km2', 'pop_density',
                                      'hap_median', 'hap_min', 'hap_max', 'hap_p95']}
    hotspots_map.get_root().html.add_child(folium.Element(
        _build_stats_legend_html(
            'High Income Homeowners Hotspots', high_stats, pollutant_name,
            'bottom: 50px; left: 630px;',
        )
    ))

    # Save
    pollutant_suffix = f"_{pollutant}" if pollutant else ""
    out_path = os.path.join(output_dir, f"{programname}{pollutant_suffix}.html")
    hotspots_map.save(out_path)
    print(f"✓ Interactive map saved: {out_path}")

    return hotspots_map


# ---------------------------------------------------------------------------
# Orchestration functions
# ---------------------------------------------------------------------------

def load_hotspot_data(popair_path, grid_path=None):
    """
    Load population+air GeoPackage and optionally the 1km air grid.

    Parameters
    ----------
    popair_path : str
        Path to the merged population+air GeoPackage produced by ip1_2cv2_popair.
    grid_path : str or None, optional
        Path to the 1km air grid GeoPackage. If None, returns None for grid_gdf.

    Returns
    -------
    tuple
        (popair_gdf, grid_gdf) where grid_gdf is None if grid_path is None.
    """
    import geopandas as gpd
    popair_gdf = gpd.read_file(popair_path)
    grid_gdf = gpd.read_file(grid_path) if grid_path is not None else None
    return popair_gdf, grid_gdf


def generate_hotspot_contours(
    popair_gdf,
    pollutant,
    demographic_groups,
    programname,
    output_dir=None,
    cell_size_m=400,
    window_radius_m=500,
    contour_increment=10,
    methods=None,
    zscore_thresholds=None,
    percentile_thresholds=None,
    save_shapefiles=True,
    save_figures=False,
):
    """
    Generate weight columns and weighted contours for each weighting method.

    Parameters
    ----------
    popair_gdf : GeoDataFrame
        Point data with population and air pollution columns.
    pollutant : str
        Pollutant column name, e.g. ``"uifl_1km_Benzene_mean"``.
    demographic_groups : dict
        Output of ``filter_demographic_groups()``.
    programname : str
        Prefix used for output file names.
    output_dir : str or None, optional
        Directory for output files. Defaults to ``programname``.
    cell_size_m : int, optional
        Raster cell size in metres (default 400).
    window_radius_m : int, optional
        KDE smoothing window in metres (default 500).
    contour_increment : int or float, optional
        People per contour interval (default 10).
    methods : list of str, optional
        Weighting methods to run. Defaults to ``['direct', 'zscore', 'percentile']``.
    zscore_thresholds : list of int, optional
        SD thresholds for z-score method. Defaults to ``[3]``.
    percentile_thresholds : list of int, optional
        Percentile thresholds for percentile method. Defaults to ``[75]``.
    save_shapefiles : bool, optional
        Whether to persist contour shapefiles (default True).
    save_figures : bool, optional
        Whether to create and save matplotlib figures (default False).

    Returns
    -------
    tuple
        ``(popair_gdf_with_weights, contour_results)`` where ``contour_results``
        maps method name → contours dict from
        ``create_weighted_population_contours()``.
    """
    if methods is None:
        methods = ['direct', 'zscore', 'percentile']
    if zscore_thresholds is None:
        zscore_thresholds = [3]
    if percentile_thresholds is None:
        percentile_thresholds = [75]
    if output_dir is None:
        output_dir = programname

    # Build weight columns
    popair_gdf = generate_all_weights_for_pollutant(
        popair_gdf,
        pollutant_column=pollutant,
        methods=methods,
        zscore_thresholds=zscore_thresholds,
        percentile_thresholds=percentile_thresholds,
        inplace=False,
    )

    # Derive the weight column names that were created
    hap_abbrev, stat = extract_hap_and_stat(pollutant)
    method_to_weight = {}
    for method in methods:
        if method == 'direct':
            method_to_weight['direct'] = build_weight_column_name(hap_abbrev, stat, 'direct')
        elif method == 'zscore':
            for thresh in zscore_thresholds:
                method_to_weight[f'zscore'] = build_weight_column_name(hap_abbrev, stat, 'zscore', thresh)
        elif method == 'percentile':
            for thresh in percentile_thresholds:
                method_to_weight[f'percentile'] = build_weight_column_name(hap_abbrev, stat, 'percentile', thresh)

    # Create contours per method
    contour_results = {}
    for method, weight_col in method_to_weight.items():
        print(f"\n{'='*70}")
        print(f"Creating contours for {method.upper()} weight method ({weight_col})")
        print(f"{'='*70}")
        contour_results[method] = create_weighted_population_contours(
            popair_gdf,
            weight_column=weight_col,
            output_dir=output_dir,
            programname=programname,
            cell_size_m=cell_size_m,
            window_radius_m=window_radius_m,
            contour_increment=contour_increment,
            demographic_groups=demographic_groups,
            save_shapefile=save_shapefiles,
            save_figure=save_figures,
            create_folium_map=False,
        )

    return popair_gdf, contour_results


def build_merged_hotspots(contour_results, popair_gdf, pollutant):
    """
    Merge overlapping contour polygons into hotspot geometries for each demographic group.

    Parameters
    ----------
    contour_results : dict
        Output of ``generate_hotspot_contours()`` — maps method name to contours dict.
    popair_gdf : GeoDataFrame
        Point data (with weight columns added by ``generate_hotspot_contours()``).
    pollutant : str
        Pollutant column name used in ``generate_hotspot_contours()``.

    Returns
    -------
    tuple
        ``(merged_low_gdf, merged_high_gdf)`` — one row per merged hotspot polygon.
    """
    hap_abbrev, stat = extract_hap_and_stat(pollutant)

    low_income_contour_sets = {}
    high_income_contour_sets = {}
    for method, result in contour_results.items():
        if method == 'direct':
            weight_col = build_weight_column_name(hap_abbrev, stat, 'direct')
        elif method == 'zscore':
            weight_col = build_weight_column_name(hap_abbrev, stat, 'zscore', 3)
        elif method == 'percentile':
            weight_col = build_weight_column_name(hap_abbrev, stat, 'percentile', 75)
        else:
            # Fallback: try to find the weight column from result keys
            weight_col = method
        low_income_contour_sets[method] = (result['low_income_contours'], weight_col)
        high_income_contour_sets[method] = (result['high_income_contours'], weight_col)

    print("=" * 100)
    print("LOW INCOME RENTERS")
    print("=" * 100)
    merged_low_gdf = merge_contours_to_hotspots(low_income_contour_sets, popair_gdf, pollutant)
    print(f"\n→ merged_low_gdf: {len(merged_low_gdf)} rows")

    print("\n" + "=" * 100)
    print("HIGH INCOME HOMEOWNERS")
    print("=" * 100)
    merged_high_gdf = merge_contours_to_hotspots(high_income_contour_sets, popair_gdf, pollutant)
    print(f"\n→ merged_high_gdf: {len(merged_high_gdf)} rows")

    return merged_low_gdf, merged_high_gdf


def save_hotspot_results(merged_low_gdf, merged_high_gdf, output_dir, programname, pollutant=None):
    """
    Concatenate low and high income hotspot GeoDataFrames and save to disk.

    Fixes the ``merged_hotspots_gdf`` variable that was undefined in the original
    notebook save cell by labelling and combining both demographic groups first.

    Parameters
    ----------
    merged_low_gdf : GeoDataFrame
        Merged hotspot polygons for low income renters.
    merged_high_gdf : GeoDataFrame
        Merged hotspot polygons for high income homeowners.
    output_dir : str
        Directory for output files.
    programname : str
        Prefix for output file names.
    pollutant : str or None, optional
        Pollutant column name (e.g. ``"uifl_1km_Benzene_mean"``). When provided,
        filenames take the form ``{programname}_{pollutant}_*``.

    Returns
    -------
    GeoDataFrame
        Combined hotspot GeoDataFrame with a ``demographic_group`` label column.
    """
    import geopandas as gpd
    import pandas as pd

    low = merged_low_gdf.copy()
    low['demographic_group'] = 'low_income_renters'
    high = merged_high_gdf.copy()
    high['demographic_group'] = 'high_income_homeowners'
    merged_hotspots_gdf = pd.concat([low, high], ignore_index=True)
    merged_hotspots_gdf = gpd.GeoDataFrame(merged_hotspots_gdf, geometry='geometry',
                                            crs=merged_low_gdf.crs)

    os.makedirs(output_dir, exist_ok=True)

    file_stem = f"{programname}_{pollutant}" if pollutant else programname

    # GeoPackage
    gpkg_path = os.path.join(output_dir, f"{file_stem}.gpkg")
    merged_hotspots_gdf.to_file(gpkg_path, layer='merged_hotspots', driver='GPKG')
    print(f"Saved GeoPackage: {gpkg_path}")

    # CSV
    csv_path = os.path.join(output_dir, f"{file_stem}.csv")
    csv_out = merged_hotspots_gdf.copy()
    csv_out['geometry_wkt'] = csv_out['geometry'].to_wkt()
    csv_out.drop(columns='geometry').to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    return merged_hotspots_gdf


def make_hotspot_map(
    popair_path,
    pollutant,
    programname,
    grid_path=None,
    output_dir=None,
    low_income_percentile=0.25,
    high_income_percentile=0.75,
    cell_size_m=400,
    window_radius_m=500,
    contour_increment=10,
    methods=None,
    zscore_thresholds=None,
    percentile_thresholds=None,
    grid_percentile=0.95,
    save_shapefiles=True,
    save_figures=False,
):
    """
    Full pipeline: load data → filter demographics → generate contours →
    merge hotspots → create interactive folium map → save results.

    Parameters
    ----------
    popair_path : str
        Path to the merged population+air GeoPackage (from ip1_2cv2_popair).
    pollutant : str
        Pollutant column name, e.g. ``"uifl_1km_Benzene_mean"``.
    programname : str
        Name used for the output directory and file prefixes.
    grid_path : str or None, optional
        Path to 1km air grid GeoPackage for map overlay. Pass None to skip.
    output_dir : str or None, optional
        Output directory. Defaults to ``programname``.
    low_income_percentile : float, optional
        Income percentile below which households are classified as low income (default 0.25).
    high_income_percentile : float, optional
        Income percentile above which households are classified as high income (default 0.75).
    cell_size_m : int, optional
        Raster cell size in metres (default 400).
    window_radius_m : int, optional
        KDE smoothing window in metres (default 500).
    contour_increment : int or float, optional
        People per contour interval (default 10).
    methods : list of str, optional
        Weighting methods. Defaults to ``['direct', 'zscore', 'percentile']``.
    zscore_thresholds : list of int, optional
        SD thresholds for z-score method. Defaults to ``[3]``.
    percentile_thresholds : list of int, optional
        Percentile thresholds for percentile method. Defaults to ``[75]``.
    grid_percentile : float, optional
        Percentile cutoff for the grid overlay layer on the map (default 0.95).
    save_shapefiles : bool, optional
        Whether to save contour shapefiles (default True).
    save_figures : bool, optional
        Whether to create and save matplotlib figures (default False).

    Returns
    -------
    folium.Map
        Interactive hotspot map. Also saved as an HTML file in ``output_dir``.
    """
    if output_dir is None:
        output_dir = programname

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    popair_gdf, grid_gdf = load_hotspot_data(popair_path, grid_path)

    # 2. Classify demographic groups
    demographic_groups = filter_demographic_groups(
        popair_gdf,
        low_income_percentile=low_income_percentile,
        high_income_percentile=high_income_percentile,
    )

    # 3. Generate weighted contours for each method
    popair_gdf, contour_results = generate_hotspot_contours(
        popair_gdf,
        pollutant=pollutant,
        demographic_groups=demographic_groups,
        programname=programname,
        output_dir=output_dir,
        cell_size_m=cell_size_m,
        window_radius_m=window_radius_m,
        contour_increment=contour_increment,
        methods=methods,
        zscore_thresholds=zscore_thresholds,
        percentile_thresholds=percentile_thresholds,
        save_shapefiles=save_shapefiles,
        save_figures=save_figures,
    )

    # 4. Merge contours into hotspot polygons
    merged_low_gdf, merged_high_gdf = build_merged_hotspots(
        contour_results, popair_gdf, pollutant
    )

    # 5. Create interactive folium map
    hotspots_map = create_merged_hotspots_map(
        merged_low_gdf=merged_low_gdf,
        merged_high_gdf=merged_high_gdf,
        popair_gdf=popair_gdf,
        pollutant=pollutant,
        programname=programname,
        output_dir=output_dir,
        grid_gdf=grid_gdf,
        grid_percentile=grid_percentile,
    )

    # 6. Save combined results
    save_hotspot_results(merged_low_gdf, merged_high_gdf, output_dir, programname, pollutant=pollutant)

    return hotspots_map

