import pandas as pd
from data_loader import load_and_process_data

def visualize_vehicle_examples():
    # 1. Get the data using our loader script
    # We load 50,000 rows to ensure we get enough variety
    df = load_and_process_data(nrows=50000)
    
    if df is None or df.empty:
        print("Dataset is empty. Check data_loader.py")
        return

    print("\n" + "="*60)
    print("VISUALIZATION: 5 RANDOM VEHICLES")
    print("="*60)
    
    # Define the columns we care about for the project
    display_cols = ['Speed_kmh', 'Road_Slope_pct', 'Ambient_Temp_C', 'Instant_Power_kW']

    # Get unique vehicles
    unique_vehicles = df['Vehicle_ID'].unique()
    
    # Loop through first 5 available vehicles
    for i, veh_id in enumerate(unique_vehicles[:5]):
        subset = df[df['Vehicle_ID'] == veh_id]
        
        # Get metadata
        v_type = subset['Type'].iloc[0]
        v_weight = subset['Weight_kg'].iloc[0]
        
        print(f"\n[{i+1}] VEHICLE ID: {veh_id} | Type: {v_type} | Weight: {v_weight:.0f} kg")
        print("-" * 60)
        
        # Sample 5 random seconds from their trip
        if len(subset) >= 5:
            sample = subset.sample(5)[display_cols]
        else:
            sample = subset[display_cols]
            
        print(sample.to_string(index=False))
        print("-" * 60)

if __name__ == "__main__":
    visualize_vehicle_examples()