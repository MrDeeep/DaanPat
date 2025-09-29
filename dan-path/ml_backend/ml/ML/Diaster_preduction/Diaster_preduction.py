# disaster_prototype.py

"""
Disaster Prediction System Prototype

This script simulates a disaster prediction workflow:
1.  Fetches population density data for a specified region.
2.  Simulates a hazard probability map using a mock AI function.
3.  Calculates the estimated population exposure based on the hazard.
4.  Generates an interactive heatmap visualization using Folium.
5.  Saves the detailed results as a GeoJSON file.

To run:
1. Install dependencies:
   pip install torch numpy pandas rasterio geopandas folium requests
2. Execute the script:
   python disaster_prototype.py
"""

import os
import logging
import requests
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import from_bounds
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# --- 1. CONFIGURATION & SETUP ---

# Region of Interest (ROI) - Centered on a coastal area in West Bengal, India for demonstration
ROI_CENTER_LAT = 21.75
ROI_CENTER_LON = 87.85
ROI_BOX_SIZE_DEG = 0.5  # The size of the box in degrees (approx 55x55 km)

# Mock Weather Data (can be replaced with real-time API call)
WEATHER_DATA = {
    'rainfall_mm': 150.0, # Heavy rainfall
    'wind_speed_kmh': 90.0   # High wind speed
}

# Model Parameters
VULNERABILITY_FACTOR = 0.3  # Configurable factor (0 to 1)

# File Paths
OUTPUT_DIR = Path("output")
POPULATION_DATA_URL = "https://hub.worldpop.org/files/wpgp_v4_res_30_sec_ppp_2020_cnst_UNadj_IND.tif"
POPULATION_TIF_PATH = Path("data/wpgp_v4_res_30_sec_ppp_2020_cnst_UNadj_IND.tif")
VISUALIZATION_HTML_PATH = OUTPUT_DIR / "hazard_map.html"
RESULTS_GEOJSON_PATH = OUTPUT_DIR / "affected_population.geojson"


def setup_environment():
    """Create necessary directories and configure logging."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    POPULATION_TIF_PATH.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "prototype.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Environment setup complete.")

# --- 2. DATA FETCHING ---

def fetch_population_data(url: str, dest_path: Path) -> None:
    """Downloads population data if not already present."""
    if dest_path.exists():
        logging.info(f"Population data already exists at {dest_path}. Skipping download.")
        return

    logging.info(f"Downloading population data from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Will stop here if the URL is broken (404 error)

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Population data saved to {dest_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download data: {e}")
        raise
# --- 3. HAZARD PREDICTION (MOCK AI MODEL) ---

def simulate_hazard_prediction(grid_shape: tuple, weather: dict) -> np.ndarray:
    """
    Simulates a hazard probability map using a mock AI function.
    
    This function creates a plausible-looking hazard map by generating random
    "hotspots" and applying a blur to simulate environmental spread.
    Real-world implementation would replace this with a trained ML model.
    
    Args:
        grid_shape (tuple): The (height, width) of the output grid.
        weather (dict): Dictionary with weather data like rainfall and wind speed.
        
    Returns:
        np.ndarray: A 2D numpy array with hazard probabilities (0-1).
    """
    logging.info("Simulating hazard prediction...")
    
    # Create a base hazard grid (tensor for torch operations)
    hazard_tensor = torch.zeros(1, 1, *grid_shape, dtype=torch.float32)

    # --- MOCK AI LOGIC ---
    # 1. Create random "epicenters" for the hazard
    num_hotspots = np.random.randint(3, 7)
    for _ in range(num_hotspots):
        x, y = np.random.randint(0, grid_shape[1]), np.random.randint(0, grid_shape[0])
        hazard_tensor[0, 0, y, x] = 1.0

    # 2. Simulate spread using a Gaussian blur (convolution)
    # A larger kernel simulates wider spread
    kernel_size = 31
    kernel = torch.from_numpy(
        np.outer(
            np.hanning(kernel_size), 
            np.hanning(kernel_size)
        )
    ).float().unsqueeze(0).unsqueeze(0)
    kernel /= kernel.sum() # Normalize
    
    # Apply convolution
    hazard_tensor = F.conv2d(hazard_tensor, kernel, padding=kernel_size // 2)

    # 3. Modulate hazard based on weather data
    # This is where external inputs would influence the model's output
    rainfall_factor = 1 + min(weather.get('rainfall_mm', 0) / 200, 1.0) # More rain, more hazard
    wind_factor = 1 + min(weather.get('wind_speed_kmh', 0) / 150, 0.5) # More wind, more hazard
    
    hazard_map = hazard_tensor.squeeze().numpy()
    hazard_map *= (rainfall_factor * wind_factor)

    # 4. Normalize to a probability range [0, 1]
    if hazard_map.max() > 0:
        hazard_map = (hazard_map - hazard_map.min()) / (hazard_map.max() - hazard_map.min())
    
    # Clip to ensure values are strictly between 0 and 1
    hazard_map = np.clip(hazard_map, 0, 1)

    logging.info("Hazard simulation complete.")
    return hazard_map

# --- 4. POPULATION EXPOSURE CALCULATION ---

def calculate_population_exposure(
    hazard_map: np.ndarray,
    population_map: np.ndarray,
    vulnerability: float
) -> np.ndarray:
    """
    Calculates the number of affected people in each grid cell.
    
    Formula: affected_people = hazard_prob * population * vulnerability
    
    Args:
        hazard_map (np.ndarray): 2D array of hazard probabilities.
        population_map (np.ndarray): 2D array of population counts.
        vulnerability (float): A factor representing susceptibility to harm.
        
    Returns:
        np.ndarray: A 2D array of estimated affected population per cell.
    """
    logging.info("Calculating population exposure...")
    if hazard_map.shape != population_map.shape:
        raise ValueError("Hazard map and population map must have the same dimensions.")
    
    affected_population_map = hazard_map * population_map * vulnerability
    # Replace negative values (often NoData values in TIFs) with 0
    affected_population_map[affected_population_map < 0] = 0
    
    total_affected = int(np.sum(affected_population_map))
    logging.info(f"Total estimated affected population: {total_affected:,}")
    return affected_population_map

# --- 5. VISUALIZATION AND REPORTING ---

def create_visualization(
    hazard_map: np.ndarray,
    affected_map: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float
) -> folium.Map:
    """
    Creates an interactive Folium map visualizing the hazard and affected population.
    
    Args:
        hazard_map (np.ndarray): Grid of hazard probabilities.
        affected_map (np.ndarray): Grid of affected population.
        transform (rasterio.Affine): Geo-transform to convert grid cells to coordinates.
        lat (float): Center latitude for the map.
        lon (float): Center longitude for the map.
        
    Returns:
        folium.Map: The generated map object.
    """
    logging.info("Creating visualization...")
    
    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    
    # Prepare data for HeatMap plugin
    heatmap_data = []
    height, width = hazard_map.shape
    for r in range(height):
        for c in range(width):
            prob = hazard_map[r, c]
            if prob > 0.05: # Threshold to avoid cluttering the map
                lon_coord, lat_coord = transform * (c, r)
                heatmap_data.append([lat_coord, lon_coord, prob])
    
    # Add HeatMap layer
    HeatMap(heatmap_data, radius=15, blur=10).add_to(m)

    # Add markers for top 10 most affected locations
    # Flatten arrays and get indices of top 10 affected cells
    flat_affected = affected_map.flatten()
    top_indices = flat_affected.argsort()[-10:][::-1] # Top 10 descending
    top_coords = np.unravel_index(top_indices, affected_map.shape)

    for i in range(len(top_coords[0])):
        r, c = top_coords[0][i], top_coords[1][i]
        affected_count = int(affected_map[r, c])
        
        if affected_count > 1: # Only show markers with significant impact
            lon_coord, lat_coord = transform * (c, r)
            popup_html = f"""
            <b>Location:</b> ({lat_coord:.4f}, {lon_coord:.4f})<br>
            <b>Hazard Probability:</b> {hazard_map[r, c]:.2%}<br>
            <b>Est. Affected People:</b> {affected_count:,}
            """
            folium.Marker(
                location=[lat_coord, lon_coord],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
    m.save(VISUALIZATION_HTML_PATH)
    logging.info(f"Visualization saved to {VISUALIZATION_HTML_PATH}")
    return m

def save_results_as_geojson(
    hazard_map: np.ndarray,
    population_map: np.ndarray,
    affected_map: np.ndarray,
    transform: rasterio.Affine
) -> None:
    """Saves the grid-cell level results to a GeoJSON file."""
    logging.info("Saving results to GeoJSON...")
    
    height, width = hazard_map.shape
    data = []

    for r in range(height):
        for c in range(width):
            lon, lat = transform * (c, r)
            population = population_map[r, c]
            if population > 0: # Only include cells with population
                data.append({
                    'hazard_prob': hazard_map[r, c],
                    'population': int(population),
                    'affected_population': int(affected_map[r, c]),
                    'geometry': Point(lon, lat)
                })

    if not data:
        logging.warning("No data to save to GeoJSON.")
        return

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    gdf.to_file(RESULTS_GEOJSON_PATH, driver='GeoJSON')
    logging.info(f"Results saved to {RESULTS_GEOJSON_PATH}")

# --- 6. MAIN WORKFLOW ---

def main():
    """Main function to run the disaster prediction pipeline."""
    setup_environment()
    logging.info("--- Starting Disaster Prediction Prototype ---")
    
    # Step 1: Get Data
    fetch_population_data(POPULATION_DATA_URL, POPULATION_TIF_PATH)
    
    # Step 2: Load and crop population data to the region of interest
    bbox = (
        ROI_CENTER_LON - ROI_BOX_SIZE_DEG / 2,
        ROI_CENTER_LAT - ROI_BOX_SIZE_DEG / 2,
        ROI_CENTER_LON + ROI_BOX_SIZE_DEG / 2,
        ROI_CENTER_LAT + ROI_BOX_SIZE_DEG / 2
    )
    
    with rasterio.open(POPULATION_TIF_PATH) as src:
        window = from_bounds(*bbox, src.transform)
        transform = src.window_transform(window)
        population_grid = src.read(1, window=window)
        # Handle NoData values, common in population datasets
        population_grid[population_grid < 0] = 0

    logging.info(f"Loaded population data for ROI. Grid shape: {population_grid.shape}")
    
    # Step 3: Simulate Hazard Prediction (This is the mock AI model part)
    # The real system would take satellite imagery or other data as input here.
    hazard_probability_map = simulate_hazard_prediction(
        grid_shape=population_grid.shape,
        weather=WEATHER_DATA
    )
    
    # Step 4: Calculate Population Exposure
    affected_population_map = calculate_population_exposure(
        hazard_map=hazard_probability_map,
        population_map=population_grid,
        vulnerability=VULNERABILITY_FACTOR
    )

    # Step 5: Visualize and Save Results
    create_visualization(
        hazard_map=hazard_probability_map,
        affected_map=affected_population_map,
        transform=transform,
        lat=ROI_CENTER_LAT,
        lon=ROI_CENTER_LON
    )
    
    save_results_as_geojson(
        hazard_map=hazard_probability_map,
        population_map=population_grid,
        affected_map=affected_population_map,
        transform=transform
    )

    logging.info("--- Prototype Run Finished Successfully ---")
    print(f"\n✅ All tasks completed. Check the '{OUTPUT_DIR}' directory for results.")
    print(f"   - Interactive Map: {VISUALIZATION_HTML_PATH}")
    print(f"   - Data Export: {RESULTS_GEOJSON_PATH}")
    print(f"   - Log File: {OUTPUT_DIR / 'prototype.log'}")


if __name__ == "__main__":
    main()