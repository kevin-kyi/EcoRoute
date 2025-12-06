import pandas as pd
import numpy as np
import os
import glob
import zipfile
import sys

# CONFIGURATION
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def print_status(msg):
    """Helper to force print output to terminal immediately"""
    print(f"[LOADER] {msg}", flush=True)

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def extract_eved_data():
    """Finds the zip file and extracts it."""
    zip_path = os.path.join(DATA_DIR, "eved-dataset", "data", "eVED.zip")
    extract_path = os.path.join(DATA_DIR, "eved_extracted")
    
    # Logic: If extract folder doesn't exist OR is empty, run unzip
    if os.path.exists(zip_path):
        if not os.path.exists(extract_path) or not os.listdir(extract_path):
            print_status(f"Unzipping eVED data to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    return extract_path

def load_and_process_data(nrows=None):
    print_status("STARTING DATA PIPELINE...")
    
    # 1. Setup & Load Registry
    setup_directories()
    extract_path = extract_eved_data()
    
    print_status("Loading Vehicle Registry...")
    static_file = os.path.join(DATA_DIR, "VED", "Data", "VED_Static_Data_PHEV&EV.xlsx")
    
    try:
        static_df = pd.read_excel(static_file)
        # Create Registry
        vehicle_registry = pd.DataFrame()
        vehicle_registry['Vehicle_ID'] = static_df['VehId']
        vehicle_registry['Weight_kg'] = static_df['Generalized_Weight'] * 0.453592
        vehicle_registry['Type'] = static_df['EngineType']
        
        valid_ev_ids = set(vehicle_registry['Vehicle_ID'].unique())
        print_status(f"-> Found {len(valid_ev_ids)} valid EV IDs in registry.")
    except Exception as e:
        print_status(f"Error loading VED Excel: {e}")
        return None

    # 2. Find Trip Data
    print_status("Finding EV Trips in CSV...")
    csv_files = glob.glob(f"{extract_path}/**/*.csv", recursive=True)
    if not csv_files:
        print_status("No CSV files found. Check 'eved_extracted' folder.")
        return None
    
    target_csv = max(csv_files, key=os.path.getsize)
    print_status(f"-> Target File: {os.path.basename(target_csv)}")

    # 3. CHUNK PROCESSING (The Sifter)
    print_status("-> Scanning file for EVs (this takes 10-20s)...")
    chunk_size = 100000
    chunks = []
    total_rows_checked = 0
    
    try:
        for i, chunk in enumerate(pd.read_csv(target_csv, chunksize=chunk_size, low_memory=False)):
            total_rows_checked += len(chunk)
            
            cols = chunk.columns
            veh_col = next((c for c in cols if 'VehId' in c), None)
            
            if veh_col:
                ev_chunk = chunk[chunk[veh_col].isin(valid_ev_ids)].copy()
                
                if not ev_chunk.empty:
                    chunks.append(ev_chunk)
                    if i % 5 == 0: print(".", end="", flush=True)
            
            if nrows and total_rows_checked >= nrows * 10: 
                break
                
        print() 
    except Exception as e:
        print_status(f"Error reading CSV chunks: {e}")
        return None

    if not chunks:
        print_status("CRITICAL: No matching EV data found in file.")
        return None

    trips_df = pd.concat(chunks)
    print_status(f"-> Aggregated {len(trips_df)} raw EV rows.")

    # 4. Standardize Column Names
    column_map = {
        'VehId': 'Vehicle_ID',
        'Vehicle Speed[km/h]': 'Speed_kmh',
        'Gradient': 'Road_Slope_pct',       
        'OAT[DegC]': 'Ambient_Temp_C',      
        'HV Battery Current[A]': 'Current_A',
        'HV Battery Voltage[V]': 'Voltage_V',
        'Timestamp(ms)': 'Timestamp_ms'
    }
    if 'Trip' in trips_df.columns: column_map['Trip'] = 'Trip_ID'
    elif 'DayNum' in trips_df.columns: column_map['DayNum'] = 'Trip_ID'
    
    trips_df = trips_df.rename(columns={k:v for k,v in column_map.items() if k in trips_df.columns})
    
    # 5. Merge & Feature Engineering
    final_df = trips_df.merge(vehicle_registry, on='Vehicle_ID', how='inner')
    
    print_status("Engineering Physics Features...")
    
    # A. FLIP SIGN: Consumption = Positive
    final_df['Instant_Power_kW'] = -1 * (final_df['Voltage_V'] * final_df['Current_A']) / 1000

    # B. SMOOTH SPEED & ACCEL
    if 'Timestamp_ms' in final_df.columns:
        final_df = final_df.sort_values(by=['Vehicle_ID', 'Trip_ID', 'Timestamp_ms'])
        
        grouped = final_df.groupby(['Vehicle_ID', 'Trip_ID'])
        
        # Smooth Speed (removes GPS noise)
        final_df['Speed_Smooth'] = grouped['Speed_kmh'].transform(
            lambda x: x.rolling(window=5, center=True).mean()
        ).fillna(final_df['Speed_kmh'])

        # Calculate Accel from Smooth Speed
        final_df['Speed_m_s'] = final_df['Speed_Smooth'] / 3.6
        final_df['Delta_Speed'] = grouped['Speed_m_s'].diff()
        final_df['Delta_Time'] = grouped['Timestamp_ms'].diff() / 1000
        
        # Calculate Accel (Handle divide by zero)
        final_df['Acceleration_m_s2'] = (final_df['Delta_Speed'] / final_df['Delta_Time']).replace([np.inf, -np.inf], 0)
    else:
        # Fallback if timestamp missing
        grouped = final_df.groupby(['Vehicle_ID', 'Trip_ID'])
        final_df['Speed_Smooth'] = final_df['Speed_kmh']
        final_df['Acceleration_m_s2'] = grouped['Speed_kmh'].diff() / 3.6

    # C. Target Smoothing
    final_df['Instant_Power_kW'] = grouped['Instant_Power_kW'].transform(
        lambda x: x.rolling(window=5, center=True).mean()
    )
    
    print_status(f"-> Rows before cleaning: {len(final_df)}")
    
    # Only drop rows if PHYSICS variables are missing.
    essential_cols = ['Speed_Smooth', 'Road_Slope_pct', 'Ambient_Temp_C', 'Weight_kg', 'Instant_Power_kW', 'Acceleration_m_s2']
    final_df = final_df.dropna(subset=essential_cols)
    
    # Filter Logic
    final_df = final_df[final_df['Speed_Smooth'] > 1] 
    final_df = final_df[
        (final_df['Instant_Power_kW'] < 250) & 
        (final_df['Instant_Power_kW'] > -100)
    ]

    print_status(f"Data Pipeline Complete. Loaded {len(final_df)} clean samples.")
    return final_df

if __name__ == "__main__":
    df = load_and_process_data(nrows=50000)
    if df is not None:
        print(df[['Speed_Smooth', 'Acceleration_m_s2', 'Instant_Power_kW']].head())