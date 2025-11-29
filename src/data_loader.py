import pandas as pd
import numpy as np
import os
import glob
import subprocess
import zipfile

# CONFIGURATION
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VED_REPO_URL = "https://github.com/gsoh/VED.git"
EVED_REPO_URL = "https://Datarepo@bitbucket.org/datarepo/eved-dataset.git"

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

def clone_repos():
    """Clones VED and eVED if they don't exist."""
    # 1. Clone VED
    ved_path = os.path.join(DATA_DIR, "VED")
    if not os.path.exists(ved_path):
        print("Cloning VED (Static Data)...")
        subprocess.run(["git", "clone", VED_REPO_URL, ved_path], check=True)
    
    # 2. Clone eVED
    eved_path = os.path.join(DATA_DIR, "eved-dataset")
    if not os.path.exists(eved_path):
        print("Cloning eVED (Dynamic Data)...")
        subprocess.run(["git", "clone", EVED_REPO_URL, eved_path], check=True)

def extract_eved_data():
    """Finds the zip file and extracts it."""
    zip_path = os.path.join(DATA_DIR, "eved-dataset", "data", "eVED.zip")
    extract_path = os.path.join(DATA_DIR, "eved_extracted")
    
    if os.path.exists(zip_path):
        if not os.path.exists(extract_path):
            print(f"Unzipping eVED data to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    else:
        print(f" Warning: Could not find {zip_path}. Check if clone was successful.")
    return extract_path

def load_and_process_data(nrows=50000):
    """Main function to load, merge, and return the training DataFrame."""
    
    # 1. Setup
    setup_directories()
    clone_repos()
    extract_path = extract_eved_data()
    
    # 2. Load Static Registry (Vehicle Weights)
    print("\nLoading Vehicle Registry...")
    static_file = os.path.join(DATA_DIR, "VED", "Data", "VED_Static_Data_PHEV&EV.xlsx")
    
    try:
        static_df = pd.read_excel(static_file)
        vehicle_registry = pd.DataFrame()
        vehicle_registry['Vehicle_ID'] = static_df['VehId']
        vehicle_registry['Weight_kg'] = static_df['Generalized_Weight'] * 0.453592
        vehicle_registry['Type'] = static_df['EngineType']
    except Exception as e:
        print(f"Error loading VED Excel: {e}")
        return None

    # 3. Load Dynamic Trip Data
    print(f"📉 Loading Trip Data (Limit: {nrows} rows)...")
    csv_files = glob.glob(f"{extract_path}/**/*.csv", recursive=True)
    
    if not csv_files:
        print("No CSV files found. Extraction might have failed.")
        return None
        
    target_csv = max(csv_files, key=os.path.getsize)
    
    # Read CSV
    try:
        trips_df = pd.read_csv(target_csv, nrows=nrows)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # 4. Map Columns (Using the EXACT names we found in validation)
    column_map = {
        'VehId': 'Vehicle_ID',
        'Vehicle Speed[km/h]': 'Speed_kmh',
        'Gradient': 'Road_Slope_pct',       
        'OAT[DegC]': 'Ambient_Temp_C',      
        'HV Battery Current[A]': 'Current_A',
        'HV Battery Voltage[V]': 'Voltage_V'
    }
    
    # Rename only columns that exist
    trips_df = trips_df.rename(columns={k:v for k,v in column_map.items() if k in trips_df.columns})
    
    # 5. Merge
    final_df = trips_df.merge(vehicle_registry, on='Vehicle_ID', how='inner')
    
    # 6. Calculate Target
    if 'Voltage_V' in final_df.columns and 'Current_A' in final_df.columns:
        final_df['Instant_Power_kW'] = (final_df['Voltage_V'] * final_df['Current_A']) / 1000
    
    print(f"Data Pipeline Complete. Loaded {len(final_df)} samples.")
    return final_df

if __name__ == "__main__":
    # Test run if executed directly
    df = load_and_process_data(nrows=1000)
    print(df.head())