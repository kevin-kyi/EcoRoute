import streamlit as st
import folium
from streamlit_folium import st_folium
import openrouteservice
import requests
import pandas as pd
import xgboost as xgb
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImNhMzQ5ZjcwOTk2MjRlYjhhODRhMDg5NmJlNDg5Nzc2IiwiaCI6Im11cm11cjY0In0="
OCM_API_KEY = "6050848c-e5f2-4368-bdf6-af9b90b26dbe"

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ev_energy_model.json")

# PITTSBURGH BOUNDING BOX
PGH_BBOX = [-80.60, 40.00, -79.30, 40.90]

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        return model
    return None

def calculate_ascent(coords):
    if not coords or len(coords[0]) < 3: return 0
    total_ascent = 0
    for i in range(1, len(coords)):
        diff = coords[i][2] - coords[i-1][2]
        if diff > 0: total_ascent += diff
    return total_ascent

def predict_energy_with_model(model, segments, weight_kg):
    df = pd.DataFrame(segments)
    df['Weight_kg'] = weight_kg
    df['Ambient_Temp_C'] = 20  
    
    features = ['Speed_Smooth', 'Road_Slope_pct', 'Ambient_Temp_C', 'Weight_kg', 'Acceleration_m_s2']
    for f in features:
        if f not in df.columns: df[f] = 0
            
    preds = model.predict(df[features])
    return (preds * df['Duration_s']).sum() / 3600

def generate_simulated_segments(duration_sec, num_segments=10):
    segments = []
    for j in range(num_segments):
        segments.append({
            'Speed_Smooth': 60 + np.random.normal(0, 5), 
            'Road_Slope_pct': np.random.normal(0, 2),    
            'Acceleration_m_s2': 0.1 if j % 2 == 0 else -0.1,
            'Duration_s': duration_sec / num_segments
        })
    return segments

def find_nearby_charger(lat, lon):
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json", "latitude": lat, "longitude": lon,
        "distance": 5, "maxresults": 1, "key": OCM_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=5).json()
        if resp:
            poi = resp[0]['AddressInfo']
            return {'lat': poi['Latitude'], 'lon': poi['Longitude'], 'title': poi['Title']}
    except:
        pass
    return None

# ==========================================
# 2. UI SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="EV Route Optimizer")

if 'map_layers' not in st.session_state: st.session_state.map_layers = []
if 'best_route_stats' not in st.session_state: st.session_state.best_route_stats = None

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚡ AI-Powered EV Route Optimizer")
    st.markdown("---")
    
    st.header("🚗 Vehicle Settings")
    battery_cap = st.number_input("Battery Capacity (kWh)", value=60)
    current_charge = st.slider("Current Charge (%)", 0, 100, 80)
    weight_kg = st.number_input("Vehicle Weight (kg)", value=1800)
    efficiency_rated = st.number_input("Rated Efficiency (kWh/km)", value=0.18, step=0.01)
    
    st.markdown("---")
    st.header("📍 Route Settings")
    start_loc = st.text_input("Start Location", "Carnegie Mellon University")
    end_loc = st.text_input("Destination", "Benedum Center")
    n_routes = st.slider("Check N Routes", 1, 3, 2)
    
    run_btn = st.button("🚀 Optimize Route", type="primary")

# ==========================================
# 4. MAIN LOGIC
# ==========================================
if run_btn:
    model = load_model()
    st.session_state.map_layers = []
    st.session_state.best_route_stats = None

    if not model:
        st.error("❌ Model not found! Please run 'train_model.py' first.")
    else:
        with st.status("Thinking...", expanded=True) as status:
            try:
                client = openrouteservice.Client(key=ORS_API_KEY)
                
                # 1. Geocode
                status.write("📍 Geocoding addresses...")
                search_start = start_loc if "Pittsburgh" in start_loc else f"{start_loc}, Pittsburgh, PA"
                search_end = end_loc if "Pittsburgh" in end_loc else f"{end_loc}, Pittsburgh, PA"
                
                start_geo = client.pelias_search(text=search_start, rect_min_x=PGH_BBOX[0], rect_min_y=PGH_BBOX[1], rect_max_x=PGH_BBOX[2], rect_max_y=PGH_BBOX[3])['features'][0]
                end_geo = client.pelias_search(text=search_end, rect_min_x=PGH_BBOX[0], rect_min_y=PGH_BBOX[1], rect_max_x=PGH_BBOX[2], rect_max_y=PGH_BBOX[3])['features'][0]
                
                start_coords = start_geo['geometry']['coordinates']
                end_coords = end_geo['geometry']['coordinates']
                st.info(f"Route: {start_geo['properties']['label']} ➝ {end_geo['properties']['label']}")
                
                # 2. Get Routes
                status.write(f"🛣️ Fetching alternatives...")
                try:
                    routes = client.directions(
                        coordinates=[start_coords, end_coords],
                        profile='driving-car', format='geojson',
                        elevation=True, # REQUEST 3D COORDINATES
                        alternative_routes={"target_count": n_routes}
                    )
                except:
                    status.write("⚠️ Switching to Single Route...")
                    routes = client.directions(
                        coordinates=[start_coords, end_coords],
                        profile='driving-car', format='geojson',
                        elevation=True
                    )
                
                # 3. Analyze Routes
                status.write("🧠 Running XGBoost Physics Model...")
                candidates = []
                best_route = None
                min_energy = float('inf')
                
                features = routes['features']
                for i, r in enumerate(features):
                    summary = r['properties']['summary']
                    dist_km = summary['distance'] / 1000
                    duration_s = summary['duration']
                    
                    coords = r['geometry']['coordinates']
                    ascent = calculate_ascent(coords)
                    
                    segments = generate_simulated_segments(duration_s)
                    pred_kwh = predict_energy_with_model(model, segments, weight_kg)
                    rated_kwh = dist_km * efficiency_rated
                    
                    candidates.append({
                        'id': i, 'kwh': pred_kwh, 'rated_kwh': rated_kwh,
                        'dist': dist_km, 'time_min': duration_s/60,
                        'ascent': ascent,
                        'geo': r, 'coords': coords
                    })
                    
                    if pred_kwh < min_energy:
                        min_energy = pred_kwh
                        best_route = candidates[-1]

                st.session_state.best_route_stats = best_route
                
                # 4. Feasibility
                current_kwh = (current_charge / 100) * battery_cap
                needed_kwh = best_route['kwh']
                
                # Map Markers
                st.session_state.map_layers.append({'marker': [start_coords[1], start_coords[0]], 'title': "Start", 'icon': 'play', 'color': 'green'})
                st.session_state.map_layers.append({'marker': [end_coords[1], end_coords[0]], 'title': "Destination", 'icon': 'flag', 'color': 'red'})

                if current_kwh > needed_kwh:
                    # SCENARIO A: Direct
                    st.session_state.map_layers.append({
                        'geo': best_route['geo'],
                        'style': {'color': 'blue', 'weight': 5},
                        'popup': f"Best Route: {best_route['dist']:.1f} km"
                    })
                else:
                    # SCENARIO B: Charging
                    mid_idx = len(best_route['coords']) // 2
                    mid_coords = best_route['coords'][mid_idx]
                    charger = find_nearby_charger(mid_coords[1], mid_coords[0])
                    
                    if charger:
                        leg1 = client.directions(coordinates=[start_coords, [charger['lon'], charger['lat']]], profile='driving-car', format='geojson', elevation=True)
                        leg2 = client.directions(coordinates=[[charger['lon'], charger['lat']], end_coords], profile='driving-car', format='geojson', elevation=True)
                        
                        leg1_sum = leg1['features'][0]['properties']['summary']
                        leg2_sum = leg2['features'][0]['properties']['summary']
                        
                        leg1_kwh = predict_energy_with_model(model, generate_simulated_segments(leg1_sum['duration']), weight_kg)
                        leg2_kwh = predict_energy_with_model(model, generate_simulated_segments(leg2_sum['duration']), weight_kg)
                        
                        l1_ascent = calculate_ascent(leg1['features'][0]['geometry']['coordinates'])
                        l2_ascent = calculate_ascent(leg2['features'][0]['geometry']['coordinates'])
                        total_ascent = l1_ascent + l2_ascent
                        
                        arrival_at_charger_kwh = current_kwh - leg1_kwh
                        target_charge_kwh = battery_cap * 0.8
                        kwh_added = target_charge_kwh - arrival_at_charger_kwh
                        arrival_soc_dest = (((battery_cap * 0.8) - leg2_kwh) / battery_cap) * 100
                        
                        best_route['charger'] = charger
                        best_route['ascent'] = total_ascent
                        best_route['leg1'] = {'dist': leg1_sum['distance']/1000, 'time': leg1_sum['duration']/60, 'kwh': leg1_kwh, 'soc': (arrival_at_charger_kwh/battery_cap)*100}
                        best_route['leg2'] = {'dist': leg2_sum['distance']/1000, 'time': leg2_sum['duration']/60, 'kwh': leg2_kwh, 'soc': arrival_soc_dest}
                        best_route['charge_metrics'] = {'added': kwh_added, 'cost': kwh_added * 0.45, 'time': 30}
                        
                        st.session_state.map_layers.append({'geo': leg1, 'style': {'color': 'red', 'weight': 4, 'dashArray': '5, 5'}, 'popup': "Leg 1"})
                        st.session_state.map_layers.append({'geo': leg2, 'style': {'color': 'blue', 'weight': 4}, 'popup': "Leg 2"})
                        st.session_state.map_layers.append({'marker': [charger['lat'], charger['lon']], 'title': charger['title'], 'icon': 'bolt', 'color': 'orange'})
                    else:
                         st.error("⚠️ No Chargers found!")
                
                status.update(label="Optimization Complete!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 5. RENDERER
# ==========================================
m = folium.Map(location=[40.4406, -79.9959], zoom_start=12)
for layer in st.session_state.map_layers:
    if 'geo' in layer: folium.GeoJson(layer['geo'], style_function=lambda x, style=layer['style']: style).add_to(m)
    if 'marker' in layer: folium.Marker(layer['marker'], popup=layer['title'], icon=folium.Icon(color=layer['color'], icon=layer['icon'])).add_to(m)

st_folium(m, width="100%", height=600)

if st.session_state.best_route_stats:
    stats = st.session_state.best_route_stats
    current_kwh = (current_charge / 100) * battery_cap
    needed_kwh = stats['kwh']
    
    st.subheader("📊 Optimization Results")
    
    diff = stats['rated_kwh'] - stats['kwh']
    if diff > 0:
        st.success(f"💡 Physics Insight: AI predicts this route is more efficient than rated! (Saved {diff:.2f} kWh)")
    else:
        st.info(f"💡 Physics Insight: AI predicts {abs(diff):.2f} kWh extra usage due to hills/traffic.")

    if current_kwh > needed_kwh:
        # SCENARIO A: Direct (Consolidated View)
        st.success("✅ Direct Route Feasible")
        
        st.markdown("### 🏁 Full Trip Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Distance", f"{stats['dist']:.1f} km")
        c2.metric("Total Time", f"{stats['time_min']:.0f} min")
        c3.metric("Elevation Gain", f"{stats['ascent']:.0f} m")
        c4.metric("Est. Cost", "$0.00")
        c5.metric("Arrival Charge", f"{((current_kwh - needed_kwh)/battery_cap)*100:.1f}%")

    elif 'charger' in stats:
        # SCENARIO B: Charging (Split View)
        st.warning(f"🛑 Charging Required at: {stats['charger']['title']}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1️⃣ Start ➝ Charger")
            st.metric("Distance", f"{stats['leg1']['dist']:.1f} km")
            st.metric("Time", f"{stats['leg1']['time']:.0f} min")
            st.metric("Arrival Charge", f"{stats['leg1']['soc']:.1f}%")
        with c2:
            st.markdown("#### ⚡ Charging Stop")
            st.info("Charge to 80%")
            st.metric("Added Energy", f"+{stats['charge_metrics']['added']:.1f} kWh")
        with c3:
            st.markdown("#### 2️⃣ Charger ➝ Dest")
            st.metric("Distance", f"{stats['leg2']['dist']:.1f} km")
            st.metric("Time", f"{stats['leg2']['time']:.0f} min")
            st.metric("Final Charge", f"{stats['leg2']['soc']:.1f}%")
            
        st.divider()
        st.subheader("🏁 Full Trip Summary")
        total_dist = stats['leg1']['dist'] + stats['leg2']['dist']
        total_time = stats['leg1']['time'] + stats['charge_metrics']['time'] + stats['leg2']['time']
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Distance", f"{total_dist:.1f} km")
        sc2.metric("Total Time", f"{total_time:.0f} min")
        sc3.metric("Elevation Gain", f"{stats['ascent']:.0f} m")
        sc4.metric("Est. Cost", f"${stats['charge_metrics']['cost']:.2f}")