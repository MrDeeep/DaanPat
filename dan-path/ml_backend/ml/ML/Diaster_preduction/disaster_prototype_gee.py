# disaster_prototype_gee.py (Complete Final Version)

import logging
from pathlib import Path

# Google Earth Engine and Geospatial Libraries
import ee
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap

# AI and Data Handling Libraries
import torch
import torch.nn.functional as F

# --- 1. CONFIGURATION ---

# Region of Interest (ROI)
ROI_CENTER_LAT = 21.75
ROI_CENTER_LON = 87.85
ROI_BOX_SIZE_DEG = 0.5

# Mock Weather Data
WEATHER_DATA = {
    'rainfall_mm': 150.0,
    'wind_speed_kmh': 90.0
}

# Model Parameters
VULNERABILITY_FACTOR = 0.3

# File Paths
OUTPUT_DIR = Path("output")
VISUALIZATION_HTML_PATH = OUTPUT_DIR / "hazard_map_gee.html"
RESULTS_GEOJSON_PATH = OUTPUT_DIR / "affected_population_gee.geojson"


# --- 2. REUSABLE FUNCTIONS (Copied from original script) ---

def setup_environment():
    """Create necessary directories and configure logging."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "prototype.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Environment setup complete.")

def simulate_hazard_prediction(grid_shape: tuple, weather: dict) -> np.ndarray:
    """Simulates a hazard probability map using a mock AI function."""
    logging.info("Simulating hazard prediction...")
    hazard_tensor = torch.zeros(1, 1, *grid_shape, dtype=torch.float32)
    num_hotspots = np.random.randint(3, 7)
    for _ in range(num_hotspots):
        x, y = np.random.randint(0, grid_shape[1]), np.random.randint(0, grid_shape[0])
        hazard_tensor[0, 0, y, x] = 1.0
    kernel_size = 31
    kernel = torch.from_numpy(np.outer(np.hanning(kernel_size), np.hanning(kernel_size))).float().unsqueeze(0).unsqueeze(0)
    kernel /= kernel.sum()
    hazard_tensor = F.conv2d(hazard_tensor, kernel, padding=kernel_size // 2)
    rainfall_factor = 1 + min(weather.get('rainfall_mm', 0) / 200, 1.0)
    wind_factor = 1 + min(weather.get('wind_speed_kmh', 0) / 150, 0.5)
    hazard_map = hazard_tensor.squeeze().numpy()
    hazard_map *= (rainfall_factor * wind_factor)
    if hazard_map.max() > 0:
        hazard_map = (hazard_map - hazard_map.min()) / (hazard_map.max() - hazard_map.min())
    hazard_map = np.clip(hazard_map, 0, 1)
    logging.info("Hazard simulation complete.")
    return hazard_map

def calculate_population_exposure(hazard_map: np.ndarray, population_map: np.ndarray, vulnerability: float) -> np.ndarray:
    """Calculates the number of affected people in each grid cell."""
    logging.info("Calculating population exposure...")
    if hazard_map.shape != population_map.shape:
        raise ValueError("Hazard map and population map must have the same dimensions.")
    affected_population_map = hazard_map * population_map * vulnerability
    affected_population_map[affected_population_map < 0] = 0
    total_affected = int(np.sum(affected_population_map))
    logging.info(f"Total estimated affected population: {total_affected:,}")
    return affected_population_map

def create_visualization(hazard_map: np.ndarray, affected_map: np.ndarray, transform: rasterio.Affine, lat: float, lon: float) -> folium.Map:
    """Creates an interactive Folium map."""
    logging.info("Creating visualization...")
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    heatmap_data = []
    height, width = hazard_map.shape
    for r in range(height):
        for c in range(width):
            prob = hazard_map[r, c]
            if prob > 0.05:
                lon_coord, lat_coord = transform * (c, r)
                heatmap_data.append([lat_coord, lon_coord, prob])
    HeatMap(heatmap_data, radius=15, blur=10).add_to(m)
    flat_affected = affected_map.flatten()
    top_indices = flat_affected.argsort()[-10:][::-1]
    top_coords = np.unravel_index(top_indices, affected_map.shape)
    for i in range(len(top_coords[0])):
        r, c = top_coords[0][i], top_coords[1][i]
        affected_count = int(affected_map[r, c])
        if affected_count > 1:
            lon_coord, lat_coord = transform * (c, r)
            popup_html = f"<b>Hazard Probability:</b> {hazard_map[r, c]:.2%}<br><b>Est. Affected People:</b> {affected_count:,}"
            folium.Marker(location=[lat_coord, lon_coord], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    m.save(VISUALIZATION_HTML_PATH)
    logging.info(f"Visualization saved to {VISUALIZATION_HTML_PATH}")
    return m

def save_results_as_geojson(hazard_map: np.ndarray, population_map: np.ndarray, affected_map: np.ndarray, transform: rasterio.Affine) -> None:
    """Saves the grid-cell level results to a GeoJSON file."""
    logging.info("Saving results to GeoJSON...")
    height, width = hazard_map.shape
    data = []
    for r in range(height):
        for c in range(width):
            population = population_map[r, c]
            if population >= 0:
                lon, lat = transform * (c, r)
                data.append({'hazard_prob': hazard_map[r, c], 'population': int(population), 'affected_population': int(affected_map[r, c]), 'geometry': Point(lon, lat)})
    if not data:
        logging.warning("No data to save to GeoJSON.")
        return
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    gdf.to_file(RESULTS_GEOJSON_PATH, driver='GeoJSON')
    logging.info(f"Results saved to {RESULTS_GEOJSON_PATH}")


# --- 3. GOOGLE EARTH ENGINE (GEE) SPECIFIC CODE ---

def get_population_data_from_gee(roi_geom):
    """
    Fetches and clips population data using the GEE API.
    Returns the data as a NumPy array and the geotransform info.
    """
    logging.info("Fetching population data from Google Earth Engine...")
    worldpop = ee.ImageCollection("WorldPop/GP/1km/ppp_UNadj").filterDate('2020-01-01', '2020-12-31').first()
    clipped_population = worldpop.clip(roi_geom)
    data = clipped_population.sampleRectangle(region=roi_geom)
    pixels = data.get('b1').getInfo()
    transform = data.get('transform').getInfo()
    population_grid = np.array(pixels)
    affine_transform = rasterio.Affine.from_gdal(*transform)
    return population_grid, affine_transform

def main():
    """Main function updated to use Google Earth Engine."""
    setup_environment()
    
    try:
        ee.Initialize()
        logging.info("Google Earth Engine initialized successfully.")
    except Exception as e:
        logging.error("Could not initialize Earth Engine. Please follow the setup steps.")
        logging.error("1. Sign up at https://signup.earthengine.google.com/")
        logging.error("2. Run 'pip install earthengine-api google-api-python-client'")
        logging.error("3. Run 'earthengine authenticate' in your terminal.")
        return

    logging.info("--- Starting GEE Disaster Prediction Prototype ---")

    min_lon = ROI_CENTER_LON - ROI_BOX_SIZE_DEG / 2
    min_lat = ROI_CENTER_LAT - ROI_BOX_SIZE_DEG / 2
    max_lon = ROI_CENTER_LON + ROI_BOX_SIZE_DEG / 2
    max_lat = ROI_CENTER_LAT + ROI_BOX_SIZE_DEG / 2
    roi_geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    try:
        population_grid, transform = get_population_data_from_gee(roi_geometry)
    except Exception as e:
        logging.error(f"Failed to fetch data from GEE: {e}")
        return

    logging.info(f"Loaded population data for ROI. Grid shape: {population_grid.shape}")

    hazard_probability_map = simulate_hazard_prediction(grid_shape=population_grid.shape, weather=WEATHER_DATA)
    affected_population_map = calculate_population_exposure(hazard_map=hazard_probability_map, population_map=population_grid, vulnerability=VULNERABILITY_FACTOR)
    create_visualization(hazard_map=hazard_probability_map, affected_map=affected_population_map, transform=transform, lat=ROI_CENTER_LAT, lon=ROI_CENTER_LON)
    save_results_as_geojson(hazard_map=hazard_probability_map, population_map=population_grid, affected_map=affected_population_map, transform=transform)

    logging.info("--- Prototype Run Finished Successfully ---")
    print(f"\n✅ All tasks completed. Check the '{OUTPUT_DIR}' directory for results.")

if __name__ == "__main__":
    main()