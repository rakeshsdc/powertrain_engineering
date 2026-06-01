import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Powertrain Gear Ratio Optimizer", layout="wide")

# 1. EMBEDDED MOTOR DATASET (Fallback/Default data from your uploaded file)
DEFAULT_MOTOR_DATA = [
    {'RPM': 108, 'Torque': 54.68}, {'RPM': 148, 'Torque': 53.23}, {'RPM': 196, 'Torque': 52.38},
    {'RPM': 249, 'Torque': 51.05}, {'RPM': 299, 'Torque': 49.7}, {'RPM': 344, 'Torque': 48.36},
    {'RPM': 404, 'Torque': 46.83}, {'RPM': 451, 'Torque': 45.82}, {'RPM': 503, 'Torque': 44.66},
    {'RPM': 550, 'Torque': 43.98}, {'RPM': 604, 'Torque': 42.46}, {'RPM': 652, 'Torque': 41.31},
    {'RPM': 697, 'Torque': 40.37}, {'RPM': 747, 'Torque': 38.97}, {'RPM': 811, 'Torque': 37.8},
    {'RPM': 854, 'Torque': 36.75}, {'RPM': 893, 'Torque': 35.97}, {'RPM': 945, 'Torque': 35.18},
    {'RPM': 1005, 'Torque': 34.32}, {'RPM': 1051, 'Torque': 33.82}, {'RPM': 1096, 'Torque': 33.08},
    {'RPM': 1152, 'Torque': 32.4}, {'RPM': 1197, 'Torque': 31.87}, {'RPM': 1250, 'Torque': 30.96},
    {'RPM': 1301, 'Torque': 30.15}, {'RPM': 1348, 'Torque': 29.11}, {'RPM': 1400, 'Torque': 28.41},
    {'RPM': 1469, 'Torque': 27.64}, {'RPM': 1514, 'Torque': 27.47}, {'RPM': 1548, 'Torque': 27.09},
    {'RPM': 1582, 'Torque': 26.58}, {'RPM': 1626, 'Torque': 26.02}, {'RPM': 1724, 'Torque': 25.3},
    {'RPM': 1757, 'Torque': 24.66}, {'RPM': 1804, 'Torque': 24.45}, {'RPM': 1893, 'Torque': 24.04},
    {'RPM': 1936, 'Torque': 23.77}, {'RPM': 2011, 'Torque': 23.25}, {'RPM': 2058, 'Torque': 23.05},
    {'RPM': 2086, 'Torque': 22.18}, {'RPM': 2193, 'Torque': 22.29}, {'RPM': 2229, 'Torque': 21.96},
    {'RPM': 2293, 'Torque': 21.08}, {'RPM': 2382, 'Torque': 20.31}, {'RPM': 2536, 'Torque': 19.58},
    {'RPM': 2623, 'Torque': 18.98}, {'RPM': 2754, 'Torque': 17.96}, {'RPM': 2874, 'Torque': 18.2},
    {'RPM': 2942, 'Torque': 17.99}, {'RPM': 3000, 'Torque': 17.8}, {'RPM': 3084, 'Torque': 17.32},
    {'RPM': 3191, 'Torque': 17.12}, {'RPM': 3252, 'Torque': 16.93}, {'RPM': 3334, 'Torque': 16.7},
    {'RPM': 3455, 'Torque': 15.88}, {'RPM': 3477, 'Torque': 16.44}, {'RPM': 3544, 'Torque': 15.62},
    {'RPM': 3591, 'Torque': 14.91}, {'RPM': 3666, 'Torque': 15.14}, {'RPM': 3699, 'Torque': 14.87},
    {'RPM': 3819, 'Torque': 14.36}, {'RPM': 3930, 'Torque': 14.06}, {'RPM': 3961, 'Torque': 14.16},
    {'RPM': 4013, 'Torque': 13.55}, {'RPM': 4060, 'Torque': 13.41}, {'RPM': 4101, 'Torque': 13.13},
    {'RPM': 4174, 'Torque': 13.33}, {'RPM': 4267, 'Torque': 12.82}, {'RPM': 4287, 'Torque': 13.05},
    {'RPM': 4377, 'Torque': 12.39}, {'RPM': 4407, 'Torque': 12.26}, {'RPM': 4438, 'Torque': 12.19},
    {'RPM': 4523, 'Torque': 12.0}, {'RPM': 4564, 'Torque': 12.0}, {'RPM': 4608, 'Torque': 11.62},
    {'RPM': 4654, 'Torque': 11.57}, {'RPM': 4686, 'Torque': 11.4}, {'RPM': 4759, 'Torque': 11.04},
    {'RPM': 4802, 'Torque': 10.55}, {'RPM': 4847, 'Torque': 10.12}, {'RPM': 4893, 'Torque': 10.51},
    {'RPM': 4947, 'Torque': 9.46}, {'RPM': 5004, 'Torque': 8.89}, {'RPM': 5051, 'Torque': 8.03},
    {'RPM': 5100, 'Torque': 6.94}, {'RPM': 5154, 'Torque': 5.03}, {'RPM': 5191, 'Torque': 4.52},
    {'RPM': 5250, 'Torque': 2.59}, {'RPM': 5296, 'Torque': 1.38}
]

st.title("🚗 Vehicle Powertrain Reduction Ratio Optimizer")
st.markdown("""
This app computes the **optimum gear reduction ratio** where the engine operates at its **peak power RPM** precisely when the vehicle reaches its aero-drag-limited top speed, while monitoring low-speed traction constraints.
""")

# 2. SIDEBAR FOR VEHICLE PARAMETERS
st.sidebar.header("🔧 Vehicle Parameters")

mass = st.sidebar.number_input("Vehicle Mass (kg)", min_value=1.0, max_value=5000.0, value=320.0, step=10.0)
r = st.sidebar.number_input("Wheel Radius (m)", min_value=0.05, max_value=1.0, value=0.29, step=0.01)
C_rr = st.sidebar.number_input("Rolling Resistance Coeff (C_rr)", min_value=0.0, max_value=0.2, value=0.045, step=0.005, format="%.3f")
eta = st.sidebar.slider("Transmission Efficiency (%)", min_value=50, max_value=100, value=90) / 100.0

st.sidebar.header("💨 Aerodynamic Drag Parameters")
C_d = st.sidebar.slider("Drag Coefficient (C_d)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
A = st.sidebar.number_input("Frontal Area (m²)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
rho = st.sidebar.number_input("Air Density (kg/m³)", min_value=0.5, max_value=1.5, value=1.225, step=0.001, format="%.3f")

st.sidebar.header("🏁 Traction Limit Parameters")
mu = st.sidebar.slider("Tire-Road Friction Coeff (μ)", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
w_dist = st.sidebar.slider("Weight on Drive Wheels (%)", min_value=10, max_value=100, value=60) / 100.0

# 3. MOTOR CURVE FILE UPLOADER
st.sidebar.header("📊 Motor Curve Data")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV (Must have 'RPM' and 'Torque' columns)", type=["csv"])

if uploaded_file is not None:
    try:
        df_motor = pd.read_csv(uploaded_file)
        df_motor = df_motor.sort_values('RPM').reset_index(drop=True)
    except Exception as e:
        st.error(f"Error parsing file, using default data instead. Error: {e}")
        df_motor = pd.DataFrame(DEFAULT_MOTOR_DATA).sort_values('RPM').reset_index(drop=True)
else:
    df_motor = pd.DataFrame(DEFAULT_MOTOR_DATA).sort_values('RPM').reset_index(drop=True)

# 4. POWERTRAIN CALCULATIONS
g = 9.81
df_motor['Power_kW'] = df_motor['Torque'] * df_motor['RPM'] * (2 * np.pi / 60) / 1000.0

# Identify Key Motor Metrics
idx_max_p = df_motor['Power_kW'].idxmax()
rpm_peak = df_motor.loc[idx_max_p, 'RPM']
p_max_kW = df_motor.loc[idx_max_p, 'Power_kW']
T_max = df_motor['Torque'].max()

# Interpolators
torque_func = interp1d(df_motor['RPM'], df_motor['Torque'], kind='linear', fill_value=(df_motor['Torque'].iloc[0], 0.0), bounds_error=False)
power_func = interp1d(df_motor['RPM'], df_motor['Power_kW'], kind='linear', fill_value=(df_motor['Power_kW'].iloc[0], 0.0), bounds_error=False)

# Solving Analytical Top Speed from Peak Power
v_space = np.linspace(0.1, 120.0, 50000)
p_req_space = (mass * g * C_rr + 0.5 * rho * C_d * A * v_space**2) * v_space / 1000.0 # kW
v_top_analytical = v_space[np.argmin(np.abs(p_req_space - p_max_kW * eta))]
G_opt_analytical = (rpm_peak * 2 * np.pi * r) / (60.0 * v_top_analytical)

# Check Traction Limits
G_traction_limit = (mu * mass * g * w_dist * r) / (T_max * eta)

# Function to simulate actual top speed achieved for a given gear ratio G
def get_top_speed_for_G(G_val):
    v_arr = np.linspace(0.1, 100.0, 2000)
    p_req = (mass * g * C_rr + 0.5 * rho * C_d * A * v_arr**2) * v_arr / 1000.0
    rpm_arr = (G_val * 60.0 * v_arr) / (2 * np.pi * r)
    p_avail = power_func(rpm_arr) * eta
    diff = p_avail - p_req
    valid_indices = np.where(diff >= 0)[0]
    if len(valid_indices) > 0:
        return v_arr[valid_indices[-1]] * 3.6 # return in km/h
    return 0.0

# Generate Plot Data
G_range = np.linspace(1.0, max(20.0, G_opt_analytical * 2), 300)
v_top_range = [get_top_speed_for_G(G) for G in G_range]

# 5. DASHBOARD LAYOUT
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="💥 Motor Peak Power", value=f"{p_max_kW:.2f} kW", delta=f"@ {int(rpm_peak)} RPM", delta_color="off")
with col2:
    st.metric(label="🎯 Optimum Reduction Ratio (G)", value=f"{G_opt_analytical:.3f}")
with col3:
    st.metric(label="🏁 Aero-Limited Top Speed", value=f"{v_top_analytical * 3.6:.2f} km/h")

st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("💡 Traction Limit Verification")
    if G_opt_analytical <= G_traction_limit:
        st.success(f"✅ **Safe Zone**: The optimum gear ratio ($G = {G_opt_analytical:.2f}$) is safely below the skidding limit ($G_{{max}} = {G_traction_limit:.2f}$). Full throttle will not break traction from a standstill.")
    else:
        st.warning(f"⚠️ **Traction Limited**: The optimum gear ratio ($G = {G_opt_analytical:.2f}$) exceeds the wheel skidding limit ($G_{{max}} = {G_traction_limit:.2f}$). Under high torque at low speeds, the car will spin its wheels.")

    st.subheader("📊 Motor Reference Snapshot")
    st.dataframe(df_motor[['RPM', 'Torque', 'Power_kW']].describe().T)

with col_right:
    st.subheader("📈 Top Speed vs. Reduction Ratio")
    
    fig = go.Figure()
    
    # Core performance curve
    fig.add_trace(go.Scatter(x=G_range, y=v_top_range, mode='lines', name='Achievable Top Speed', line=dict(color='#1f77b4', width=3)))
    
    # Optimum Marker
    fig.add_trace(go.Scatter(x=[G_opt_analytical], y=[v_top_analytical * 3.6], mode='markers+text', 
                             name='Optimum Ratio', text=[f" Optimum G: {G_opt_analytical:.2f}"],
                             textposition="top right", marker=dict(color='green', size=12, symbol='star')))
    
    # Traction Limit Shading / Line
    fig.add_vline(x=G_traction_limit, line_dash="dash", line_color="red", 
                  annotation_text=f"Traction Skidding Limit (G = {G_traction_limit:.2f})", annotation_position="bottom right")
    
    fig.update_layout(
        xaxis_title="Reduction Ratio (G) [X-Axis]",
        yaxis_title="Vehicle Top Speed (km/h) [Y-Axis]",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
