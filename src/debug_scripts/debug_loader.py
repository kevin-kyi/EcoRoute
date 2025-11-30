import pandas as pd
import numpy as np
import os
import glob
import zipfile

# CONFIGURATION
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def extract_eved_data():
    """Finds the zip file and extracts it."""
    zip_path = os.path.join(DATA_DIR, "eved-dataset", "data", "eVED.zip")
    extract_path = os.path.join(DATA_DIR, "eved_extracted")
    if os.path.exists(zip_path) and not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    return extract_path

def debug_load():
    print("🐞 STARTING DEBUG LOADER")
    
    # 1. SETUP
    extract_path = extract_eved_data()
    static_file = os.path.join(DATA_DIR, "VED", "Data", "VED_Static_Data_PHEV&EV.xlsx")
    
    # Load Registry
    static_df = pd.read_excel(static_file)
    vehicle_registry = pd.DataFrame({
        'Vehicle_ID': static_df['VehId'],
        'Type': static_df['EngineType']
    })
    valid_ev_ids = set(vehicle_registry['Vehicle_ID'].unique())
    
    # Find CSV
    csv_files = glob.glob(f"{extract_path}/**/*.csv", recursive=True)
    target_csv = max(csv_files, key=os.path.getsize)
    print(f"   -> Reading target: {os.path.basename(target_csv)}")

    # 2. LOAD RAW CHUNKS (Just like before)
    chunks = []
    for chunk in pd.read_csv(target_csv, chunksize=100000, low_memory=False):
        cols = chunk.columns
        veh_col = next((c for c in cols if 'VehId' in c), None)
        if veh_col:
            ev_chunk = chunk[chunk[veh_col].isin(valid_ev_ids)].copy()
            if not ev_chunk.empty:
                chunks.append(ev_chunk)
    
    if not chunks:
        print("❌ No EV rows found.")
        return

    df = pd.concat(chunks)
    print(f"   -> 📊 Raw Merged Rows: {len(df)}")

    # 3. CHECKPOINT A: COLUMN MAPPING
    column_map = {
        'VehId': 'Vehicle_ID',
        'Vehicle Speed[km/h]': 'Speed_kmh',
        'Gradient': 'Road_Slope_pct',       
        'OAT[DegC]': 'Ambient_Temp_C',      
        'HV Battery Current[A]': 'Current_A',
        'HV Battery Voltage[V]': 'Voltage_V',
        'Timestamp(ms)': 'Timestamp_ms'
    }
    # Fix Trip ID
    if 'Trip' in df.columns: column_map['Trip'] = 'Trip_ID'
    elif 'DayNum' in df.columns: column_map['DayNum'] = 'Trip_ID'
        
    df = df.rename(columns={k:v for k,v in column_map.items() if k in df.columns})
    
    # 3. CHECKPOINT B: POWER CALCULATION
    print("\n🔍 Checking Power Calculation...")
    if 'Voltage_V' in df.columns and 'Current_A' in df.columns:
        df['Instant_Power_kW'] = (df['Voltage_V'] * df['Current_A']) / 1000
        missing_power = df['Instant_Power_kW'].isna().sum()
        print(f"   -> Rows with NaN Power: {missing_power}")
        print(f"   -> Sample Power: {df['Instant_Power_kW'].head(3).tolist()}")
    else:
        print("   ❌ MISSING VOLTAGE OR CURRENT COLUMNS!")
        print(f"   -> Columns available: {df.columns.tolist()}")

    # 4. CHECKPOINT C: ACCELERATION & TIMESTAMP
    print("\n🔍 Checking Acceleration...")
    if 'Timestamp_ms' in df.columns:
        df = df.sort_values(by=['Vehicle_ID', 'Trip_ID', 'Timestamp_ms'])
        grouped = df.groupby(['Vehicle_ID', 'Trip_ID'])
        
        df['Delta_Time'] = grouped['Timestamp_ms'].diff() / 1000
        df['Delta_Speed'] = grouped['Speed_kmh'].diff() / 3.6
        df['Acceleration_m_s2'] = df['Delta_Speed'] / df['Delta_Time']
        
        # Check for bad time deltas
        zeros = (df['Delta_Time'] == 0).sum()
        nans = df['Delta_Time'].isna().sum()
        print(f"   -> Rows with 0s Time Delta (Duplicate Timestamp): {zeros}")
        print(f"   -> Rows with NaN Time Delta (Start of Trip): {nans}")
    else:
        print("   ❌ Missing Timestamp column")

    # 5. CHECKPOINT D: THE FILTERS (The Suspects)
    print("\n🔍 Testing Filters (Where did the data go?)...")
    
    # Filter 1: DropNa
    n_before = len(df)
    df_no_na = df.dropna()
    print(f"   -> After dropna(): {len(df_no_na)} rows (Lost {n_before - len(df_no_na)})")
    
    # Filter 2: Speed > 1
    n_moving = df_no_na[df_no_na['Speed_kmh'] > 1]
    print(f"   -> After Speed > 1 filter: {len(n_moving)} rows (Lost {len(df_no_na) - len(n_moving)})")
    if len(n_moving) == 0:
        print(f"      ⚠️ AVG SPEED CHECK: {df['Speed_kmh'].mean()}")

    # Filter 3: Power Range
    n_valid_power = n_moving[
        (n_moving['Instant_Power_kW'] < 200) & 
        (n_moving['Instant_Power_kW'] > -100)
    ]
    print(f"   -> After Power Range filter: {len(n_valid_power)} rows")

    print("\n🏁 DEBUG COMPLETE")

if __name__ == "__main__":
    debug_load()