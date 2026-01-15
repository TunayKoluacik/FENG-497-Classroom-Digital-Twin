import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

st.set_page_config(layout="wide", page_title="FENG 497: Digital Twin", page_icon="🏫")

st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 32px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 20px;
    }
    .status-banner-safe {
        background: linear-gradient(90deg, #1d976c 0%, #93f9b9 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-banner-danger {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .recommendation-box {
        background-color: #2b2d42;
        border-left: 5px solid #ef233c;
        padding: 15px;
        margin-top: 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('classroom_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_resources()

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Temp', 'CO2', 'Light', 'Noise'])

def update_history(new_data):
    now = datetime.now().strftime("%H:%M:%S")
    new_row = {
        'Time': now, 
        'Temp': new_data['air_temp'],
        'CO2': new_data['co2'],
        'Light': new_data['light'],
        'Noise': new_data['noise']
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history.iloc[1:]


def generate_preset_data(zone):
    """Generates random data based on zone scenarios."""
    if zone == "Window (Zone A)":
        return {
            "air_temp": round(random.uniform(28.0, 31.0), 1),
            "op_temp_offset": random.uniform(2.0, 4.0),
            "hum": round(random.uniform(35, 50), 1),
            "light": int(random.uniform(850, 1200)),
            "noise": int(random.uniform(45, 60)),
            "co2": int(random.uniform(450, 600))
        }
    elif zone == "Back Corner (Zone B)":
        return {
            "air_temp": round(random.uniform(23.0, 25.0), 1),
            "op_temp_offset": random.uniform(0.0, 0.5),
            "hum": round(random.uniform(50, 60), 1),
            "light": int(random.uniform(200, 300)),
            "noise": int(random.uniform(30, 45)),
            "co2": int(random.uniform(1200, 1800))
        }
    else: # Center (Zone C)
        return {
            "air_temp": round(random.uniform(23.0, 25.5), 1),
            "op_temp_offset": random.uniform(0.0, 0.5),
            "hum": round(random.uniform(45, 55), 1),
            "light": int(random.uniform(450, 600)),
            "noise": int(random.uniform(40, 55)),
            "co2": int(random.uniform(500, 750))
        }

def draw_radar_chart(data):
    """Draws a spider plot to visualize balance."""
    categories = ['Temp', 'Humidity', 'Light', 'Noise', 'CO2']
    
    values = [
        min(1.0, data['air_temp'] / 35),
        min(1.0, data['hum'] / 80),
        min(1.0, data['light'] / 1500),
        min(1.0, data['noise'] / 90),
        min(1.0, data['co2'] / 2000)
    ]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#00CC96')
    ax.fill(angles, values, '#00CC96', alpha=0.4)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color='white')
    ax.set_yticks([]) # Hide radial ticks
    
    # Add an "Ideal" circle
    ideal_values = [0.7, 0.6, 0.4, 0.5, 0.3] # Rough ideal ratios
    ideal_values += ideal_values[:1]
    ax.plot(angles, ideal_values, linewidth=1, linestyle='--', color='gray', alpha=0.5)
    
    return fig

def draw_classroom_map(active_zone, is_manual=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Room
    room = patches.Rectangle((0, 0), 10, 8, linewidth=3, edgecolor='#555', facecolor='#262730')
    ax.add_patch(room)
    
    # Zones
    zones = {
        "Window (Zone A)": (0, 5, 10, 3, '#FFA500', "WINDOW ZONE"),
        "Back Corner (Zone B)": (7, 0, 3, 4, '#808080', "BACK CORNER"),
        "Center (Zone C)": (3, 2, 4, 3, '#ADD8E6', "CENTER ZONE")
    }
    
    for z_name, (x, y, w, h, c, lbl) in zones.items():
        if is_manual:
            alpha = 0.2
            edge = 'gray'
            thick = 1
        else:
            is_active = (z_name == active_zone)
            alpha = 0.7 if is_active else 0.1
            edge = 'white' if is_active else 'gray'
            thick = 2 if is_active else 1
            
        rect = patches.Rectangle((x, y), w, h, linewidth=thick, edgecolor=edge, facecolor=c, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, lbl, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.text(5, 8.2, "☀️ SUNLIGHT / WINDOWS", ha='center', fontsize=9, color='orange', fontweight='bold')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    return fig

# -SIDEBAR

st.sidebar.title("🎛️ Command Center")
mode = st.sidebar.radio("Input Source:", ["📍 Preset Zones", "🔧 Manual Override"])

# "Connect" Simulation
if st.sidebar.button("📡 Reconnect Sensors"):
    with st.sidebar:
        with st.spinner("Handshaking with Zigbee Mesh..."):
            time.sleep(1.5)
        st.success("Connected: Node ID #4X-99")

data = {}
selected_zone_label = "Manual Mode"

if mode == "📍 Preset Zones":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Scenario")
    selected_zone_label = st.sidebar.selectbox("Active Node", ["Center (Zone C)", "Window (Zone A)", "Back Corner (Zone B)"])
    if st.sidebar.button("🔄 Trigger New Reading"):
        st.rerun()  
    data = generate_preset_data(selected_zone_label)

else: # Manual Mode
    st.sidebar.markdown("---")
    air_temp = st.sidebar.slider("🌡️ Air Temp (°C)", 15.0, 40.0, 24.0, 0.5)
    rad_offset = st.sidebar.slider("☀️ Radiant Offset (°C)", 0.0, 10.0, 0.5, 0.5)
    hum = st.sidebar.slider("💧 Humidity (%)", 0, 100, 50)
    co2 = st.sidebar.slider("☁️ CO2 (ppm)", 400, 3000, 600, 50)
    light = st.sidebar.slider("💡 Light (lux)", 0, 2500, 500, 50)
    noise = st.sidebar.slider("🔊 Noise (dB)", 30, 100, 45, 5)
    data = {"air_temp": air_temp, "op_temp_offset": rad_offset, "hum": hum, "co2": co2, "light": light, "noise": noise}

# --- LOGIC ENGINE ---
op_temp = round(data['air_temp'] + data['op_temp_offset'], 1)

# AI Prediction
ai_status = "Unknown"
ai_conf = 0.0
if model and scaler:
    feats = np.array([[data['air_temp'], op_temp, data['hum']]])
    scaled = scaler.transform(feats)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]
    ai_status = "Suitable" if pred == 1 else "Unsuitable"
    ai_conf = prob[pred] * 100

# Safety Layer
override_reasons = []
actions = []
if data['co2'] > 1000: 
    override_reasons.append(f"CO2 Critical ({data['co2']} ppm)")
    actions.append("ACTIVATE HVAC VENTILATION")
if data['noise'] > 75: 
    override_reasons.append(f"Noise High ({data['noise']} dB)")
    actions.append("REDUCE VOLUME / CLOSE THE DOOR OR BLINDS")
if data['light'] > 2000: 
    override_reasons.append("Glare Detected")
    actions.append("LOWER BLINDS")
if data['light'] < 100: 
    override_reasons.append("Insufficient Light")
    actions.append("INCREASE ARTIFICIAL LIGHTING")
if ai_status == "Unsuitable" and not override_reasons:
    actions.append("ADJUST THERMOSTAT")

final_status = "Suitable" if (ai_status == "Suitable" and not override_reasons) else "Unsuitable"

# Update History
update_history(data)

# --- MAIN DASHBOARD UI ---

st.title("🏫 Intelligent Classroom Digital Twin")

# 1. HEADS UP DISPLAY (HUD)
hud_col1, hud_col2 = st.columns([3, 1])

with hud_col1:
    if final_status == "Suitable":
        st.markdown(f"""
        <div class="status-banner-safe">
            <h2 style='margin:0'>✅ LEARNING CONDITIONS OPTIMAL</h2>
            <p style='margin:0; opacity:0.9; font-size:18px'>AI Confidence: {ai_conf:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        reason_text = " + ".join(override_reasons) if override_reasons else "Thermal Discomfort (AI Model)"
        st.markdown(f"""
        <div class="status-banner-danger">
            <h2 style='margin:0'>❌ UNSUITABLE CONDITIONS</h2>
            <p style='margin:0; opacity:0.9; font-size:18px'>CAUSE: {reason_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation Box
        if actions:
            action_text = " | ".join(actions)
            st.markdown(f"""
            <div class="recommendation-box">
                <b>🛠️ RECOMMENDED ACTION:</b> {action_text}
            </div>
            """, unsafe_allow_html=True)

with hud_col2:
    st.metric("Operative Temp", f"{op_temp} °C", delta=f"{data['op_temp_offset']}°C Radiant")

st.markdown("---")

# 2. MAIN VISUALIZATION GRID
col_radar, col_graphs, col_map = st.columns([1, 1.5, 1.2])

with col_radar:
    st.subheader("📊 Env. Profile")
    st.pyplot(draw_radar_chart(data))
    st.caption("Shape Analysis: Circle = Balanced")

with col_graphs:
    st.subheader("📈 Live Sensor Trends")
    # Plotting the history
    if not st.session_state.history.empty:
        chart_data = st.session_state.history[['Temp', 'CO2', 'Light']].copy()
        # Simple normalization for display on same chart
        chart_data['CO2'] = chart_data['CO2'] / 100
        chart_data['Light'] = chart_data['Light'] / 50
        st.line_chart(chart_data)
        st.caption("Normalized Scale: Temp(x1), CO2(x100), Light(x50)")

with col_map:
    st.subheader("🗺️ Zone Status")
    st.pyplot(draw_classroom_map(selected_zone_label, is_manual=(mode=="🔧 Manual Override")))

# 3. FOOTER METRICS
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🌡️ Air Temp", f"{data['air_temp']} °C")
m2.metric("💧 Humidity", f"{data['hum']} %")
m3.metric("☁️ CO2", f"{data['co2']} ppm", delta_color="inverse")
m4.metric("💡 Light", f"{data['light']} lx")
m5.metric("🔊 Noise", f"{data['noise']} dB", delta_color="inverse")