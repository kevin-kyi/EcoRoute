import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_and_process_data

# 1. Load Data
print("Loading data for analysis...")
df = load_and_process_data(nrows=50000)

# 2. Check Correlations
# We want to see strong numbers (>0.5) for Speed and Acceleration vs Power
corr = df[['Speed_kmh', 'Acceleration_m_s2', 'Road_Slope_pct', 'Instant_Power_kW']].corr()
print("\n📊 CORRELATION MATRIX (Closer to 1.0 or -1.0 is good):")
print(corr['Instant_Power_kW'].sort_values(ascending=False))

# 3. Visual Diagnosis
plt.figure(figsize=(12, 5))

# Plot A: The "Cloud"
plt.subplot(1, 2, 1)
plt.scatter(df['Acceleration_m_s2'][:1000], df['Instant_Power_kW'][:1000], alpha=0.3)
plt.xlabel("Acceleration (m/s²)")
plt.ylabel("Power (kW)")
plt.title("Acceleration vs Power (Ideally a clean diagonal line)")

# Plot B: The "Timeline"
plt.subplot(1, 2, 2)
subset = df[df['Trip_ID'] == df['Trip_ID'].iloc[0]].iloc[:60] # First 60 seconds of a trip
plt.plot(subset['Instant_Power_kW'], label='Power (Target)')
plt.plot(subset['Acceleration_m_s2'] * 10, label='Accel (x10)') # Scaled to fit
plt.legend()
plt.title("Timeline Check: Do they move together?")

plt.tight_layout()
plt.show()