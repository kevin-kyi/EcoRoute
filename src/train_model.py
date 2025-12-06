import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Import your data loader
from data_loader import load_and_process_data

# CONFIGURATION
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_ev_model():
    print("STARTING MODEL TRAINING PIPELINE")
    print("-" * 40)

    # 1. LOAD DATA 
    df = load_and_process_data(nrows=200000)
    
    if df is None or df.empty:
        print("Error: No data loaded.")
        return

    # 2. DEFINE FEATURES & TARGET
    features = [
        'Speed_Smooth',     
        'Road_Slope_pct', 
        'Ambient_Temp_C', 
        'Weight_kg', 
        'Acceleration_m_s2'
    ]
    target = 'Instant_Power_kW'
    
    X = df[features]
    y = df[target]

    print(f"\nTraining on {len(X)} samples with features: {features}")

    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TRAIN XGBOOST MODEL
    print("\nTraining XGBoost Regressor...")
    
    model = xgb.XGBRegressor(
        n_estimators=3000,       
        learning_rate=0.015,     
        max_depth=10,            
        subsample=0.75,          
        colsample_bytree=0.8,   
        n_jobs=-1,              
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 4. EVALUATE PERFORMANCE
    print("\nEvaluation Results:")
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"   -> Mean Absolute Error: {mae:.3f} kW")
    print(f"   -> R² Score:            {r2:.3f}")
    
    if r2 > 0.6:
        print("   SUCCESS: Model is predictive.")
    else:
        print("   WARNING: Score is low. Check if 'Acceleration' logic is working.")

    # 5. SAVE THE MODEL
    model_path = os.path.join(MODEL_DIR, "ev_energy_model.json")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    plt.figure(figsize=(10, 6))
    # Plot just 100 points so the chart is readable
    plt.scatter(y_test[:100], preds[:100], alpha=0.6, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Line')
    plt.xlabel("Actual Energy (kW)")
    plt.ylabel("Predicted Energy (kW)")
    plt.title(f"Model Accuracy (R2: {r2:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(MODEL_DIR, "accuracy_plot.png")
    plt.savefig(plot_path)
    print(f"   -> Accuracy plot saved to '{plot_path}'")

if __name__ == "__main__":
    train_ev_model()