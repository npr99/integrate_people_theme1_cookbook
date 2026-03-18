# Script to update the hotspot analysis functions with contour_increment support

import re

# Read the current file
with open('ip1_0hv1_hotspot_analysis.py', 'r') as f:
    content = f.read()

# Replace the function signature for create_population_contours
old_sig1 = r'''def create_population_contours\(
    gdf,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS\["cell_size_m"\],
    window_radius_m=DEFAULT_CONTOUR_PARAMS\["window_radius_m"\],
    levels=DEFAULT_CONTOUR_PARAMS\["levels"\],
    save_shapefile=True,
\):'''

new_sig1 = '''def create_population_contours(
    gdf,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    contour_increment=None,
    levels=None,
    save_shapefile=True,
):'''

content = re.sub(old_sig1, new_sig1, content)

# Replace docstring for create_population_contours - parameters section
old_doc1 = r'''    levels : int, optional
        Number of contour levels \(default: 12\)\.'''

new_doc1 = '''    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for every 100 people).
        If provided, levels are calculated from raster data range.
        If not provided, uses 'levels' parameter (default: None).
    levels : int or list, optional
        Number of contour levels (int) or specific level values (list/array).
        Default: 12 if contour_increment is None.'''

content = re.sub(old_doc1, new_doc1, content)

# Replace the level calculation section for create_population_contours
old_calc1 = r'''    # Create figure with contours
    print\(f"Plotting \{levels\} contour levels..."\)'''

new_calc1 = '''    # Calculate contour levels if using increment
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
    
    # Create figure with contours'''

content = re.sub(old_calc1, new_calc1, content)

# Now update create_demographic_contours similarly
old_sig2 = r'''def create_demographic_contours\(
    gdf,
    demographic_groups,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS\["cell_size_m"\],
    window_radius_m=DEFAULT_CONTOUR_PARAMS\["window_radius_m"\],
    levels=DEFAULT_CONTOUR_PARAMS\["levels"\],
    save_shapefile=True,
\):'''

new_sig2 = '''def create_demographic_contours(
    gdf,
    demographic_groups,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    contour_increment=None,
    levels=None,
    save_shapefile=True,
):'''

content = re.sub(old_sig2, new_sig2, content)

# Update docstring for create_demographic_contours
old_doc2 = r'''    levels : int, optional
        Number of contour levels \(default: 12\)\.'''

new_doc2 = '''    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for every 100 people).
        If provided, levels are calculated from raster data range.
        If not provided, uses 'levels' parameter (default: None).
    levels : int or list, optional
        Number of contour levels (int) or specific level values (list/array).
        Default: 12 if contour_increment is None.'''

# This needs to be more specific
content = content.replace(
    '''    levels : int, optional
        Number of contour levels (default: 12).
    save_shapefile : bool, optional
        Whether to save contour shapefiles (default: True).

    Returns
    -------
    dict
        Dictionary with raster results and contours for both groups.
    """
    low_gdf = demographic_groups["low_income_renters"]''',
    '''    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for every 100 people).
        If provided, levels are calculated from raster data range.
        If not provided, uses 'levels' parameter (default: None).
    levels : int or list, optional
        Number of contour levels (int) or specific level values (list/array).
        Default: 12 if contour_increment is None.
    save_shapefile : bool, optional
        Whether to save contour shapefiles (default: True).

    Returns
    -------
    dict
        Dictionary with raster results and contours for both groups.
    """
    low_gdf = demographic_groups["low_income_renters"]'''
)

# Update create_weighted_population_contours similarly
content = content.replace(
    '''def create_weighted_population_contours(
    gdf,
    pollutant_column,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    levels=DEFAULT_CONTOUR_PARAMS["levels"],
    save_shapefile=True,
):''',
    '''def create_weighted_population_contours(
    gdf,
    pollutant_column,
    output_dir,
    programname,
    cell_size_m=DEFAULT_CONTOUR_PARAMS["cell_size_m"],
    window_radius_m=DEFAULT_CONTOUR_PARAMS["window_radius_m"],
    contour_increment=None,
    levels=None,
    save_shapefile=True,
):'''
)

# Update create_weighted_population_contours docstring
content = content.replace(
    '''    levels : int, optional
        Number of contour levels (default: 12).
    save_shapefile : bool, optional
        Whether to save contour shapefile (default: True).

    Returns
    -------
    dict
        Dictionary with 'raster_result' and 'contours_gdf' (if shapefile saved).
    """
    # Extract pollutant name from column for display''',
    '''    contour_increment : float or int, optional
        Increment between contour lines (e.g., 100 for weighted exposure).
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
    # Extract pollutant name from column for display'''
)

# Write updated content back
with open('ip1_0hv1_hotspot_analysis.py', 'w') as f:
    f.write(content)

print("Hotspot analysis file updated successfully!")
print("Changes made:")
print("- Added contour_increment parameter to all three create_*_contours functions")
print("- Updated docstrings to document the new parameter")
print("- Added logic to calculate contour levels based on increment and raster max values")
