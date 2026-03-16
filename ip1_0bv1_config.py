"""
Project configuration and shared constants.
"""

# Mapping for species codes to full names
SPECIES_MAPPING = {
    "antl": "Acetonitrile",
    "be13": "1,3-Butadiene",
    "benz": "Benzene",
    "etbz": "Ethylbenzene",
    "etox": "Ethylene Oxide",
    "nhex": "Hexane",
    "strn": "Styrene",
    "tolu": "Toluene",
    "xyln": "Xylenes",
}

# Texas cities coordinates for geographic context
TEXAS_CITIES = {
    "Houston": {"lat": 29.7604, "lon": -95.3698},
    "Beaumont": {"lat": 30.0860, "lon": -94.1265},
    "Port Arthur": {"lat": 29.8849, "lon": -93.9300},
    "Orange": {"lat": 30.0927, "lon": -93.7565},
    "Bridge City": {"lat": 30.0254, "lon": -93.8432},
    "Lufkin": {"lat": 31.3382, "lon": -94.7291},
    "Tyler": {"lat": 32.3513, "lon": -95.3011},
}

# Default air pollution data source
AIR_POLLUTION_URL = (
    "https://ckan.tacc.utexas.edu/dataset/"
    "c3c1b2b2-4d43-4603-b1c5-fc0877c279ae/"
    "resource/465209c8-f407-4eb2-99b3-4fc719e168c3/"
    "download/simulated_percentile_conc.zip"
)

# Hotspot analysis configuration
DEFAULT_CONTOUR_PARAMS = {
    "cell_size_m": 400,
    "window_radius_m": 500,
    "levels": 12,
}

# Demographic thresholds (percentiles)
DEFAULT_LOW_INCOME_PERCENTILE = 0.25
DEFAULT_HIGH_INCOME_PERCENTILE = 0.75

# Group quarters facility types to include in low-income analysis
# 1 = Correctional facilities for adults
# 2 = Juvenile facilities
# 3 = Nursing facilities/Skilled-nursing facilities
DEFAULT_GROUP_QUARTERS_CODES = [1, 2, 3]

# Colors for demographic visualizations
DEMOGRAPHIC_COLORS = {
    "low_income_renters": "blue",
    "high_income_homeowners": "red",
}

# HAP abbreviations for weighted population column naming
# Maps full HAP names to single-letter abbreviations
HAP_ABBREVIATIONS = {
    "Benzene": "B",
    "Toluene": "T",
    "Ethylene Oxide": "E",
    "Xylenes": "X",
    "Ethylbenzene": "EB",
    "Styrene": "S",
    "Hexane": "H",
    "1,3-Butadiene": "BD",
    "Acetonitrile": "A",
}

# Reverse mapping: abbreviation to full name
HAP_ABBREVIATIONS_REVERSE = {v: k for k, v in HAP_ABBREVIATIONS.items()}

# Weighting method names for column naming
WEIGHT_METHOD_SUFFIXES = {
    "direct": "",  # No suffix for direct
    "zscore": lambda threshold: f"z{int(threshold)}",  # e.g., z3, z2
    "percentile": lambda threshold: f"p{int(threshold)}",  # e.g., p75, p50
}
