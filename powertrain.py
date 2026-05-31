mport streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- STYLING AND TITLE ---
st.set_page_config(page_title="Dynamic Vehicle Performance Calculator", layout="wide")
st.title("Velocity Calculation with Torque-RPM Curve")
st.markdown("""
This tool calculates the maximum velocity and optimum reduction ratio by importing a 
**Torque vs. RPM** curve from an Excel file.
""")

# --- SIDEBAR: VEHICLE PARAMETERS (From Source [1]) ---
st.sidebar.header("Vehicle Parameters")
mass = st.sidebar.number_input("Vehicle Mass (kg)", value=320.0) # [1]
r = st.sidebar.number_input("Wheel Radius (m)", value=0.291) # [1]
crr = st.sidebar.number_input("Coeff. of Rolling Resistance (Crr)", value=0.045) # [1]
efficiency = st.sidebar.number_input("Transmission Efficiency", value=0.90) # [1]

st.sidebar.subheader("Environment Settings")
mu = st.sidebar.slider("Coefficient of Friction (mu)", 0.1, 1.2, 0.8)
rho = 1.225
cd = 0.30
area = 1.5
g = 9.81

# --- DATA UPLOAD ---
st.subheader("1. Upload Torque-RPM Data")
uploaded_file = st.file_uploader("Upload your .xlsx file (Must have 'RPM' and 'Torque' columns)", type=["xlsx"])

if uploaded_file:
    # Load Data
    df = pd.read_excel(uploaded_file)
    
    if 'RPM' in df.columns and 'Torque' in df.columns:
        st.success("Data loaded successfully!")
        # Sort by RPM to ensure proper interpolation
        df = df.sort_values(by='RPM')
        rpm_data = df['RPM'].values
        torque_data = df['Torque'].values
        
        # Create interpolation function for Torque(RPM)
        torque_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
        
        # --- CALCULATIONS ---
        
        def get_resistive_power(v):
            f_rr = mass * g * crr
            f_ad = 0.5 * rho * cd * area * (v**2)
            return (f_rr + f_ad) * v

        reduction_ratios = np.linspace(1, 15, 100)
        max_velocities = []
        skid_status = []

        for G in reduction_ratios:
            # Check for skidding at maximum torque point in the data
            max_tractive_force = (max(torque_data) * G * efficiency) / r
            traction_limit = mu * mass * g
            skid_status.append(max_tractive_force > traction_limit)
            
            # Find the intersection of P_available and P_required
            # We iterate through the RPM range for this G
            v_terminal = 0
            for current_rpm in np.linspace(min(rpm_data), max(rpm_data), 100):
                v_at_rpm = (2 * np.pi * current_rpm * r) / (60 * G)
                p_avail = (2 * np.pi * current_rpm * torque_func(current_rpm) * efficiency) / 60
                p_req = get_resistive_power(v_at_rpm)
                
                if p_avail >= p_req:
                    v_terminal = max(v_terminal, v_at_rpm)
                else:
                    # Power limited at this specific speed
                    break
            max_velocities.append(v_terminal)

        # --- PLOTTING ---
        st.subheader("2. Performance Analysis: Velocity vs. Reduction Ratio")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(reduction_ratios, max_velocities, color='blue', label='Max Velocity')
        
        # Highlight Skidding Zone
        skid_mask = np.array(skid_status)
        ax.fill_between(reduction_ratios, 0, max(max_velocities), where=skid_mask, 
                        color='red', alpha=0.3, label="Traction Limited (Skids)")
        
        ax.set_xlabel("Reduction Ratio (G)")
        ax.set_ylabel("Maximum Velocity (m/s)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --- RESULTS ---
        opt_v = max(max_velocities)
        opt_g = reduction_ratios[np.argmax(max_velocities)]
        
        c1, c2 = st.columns(2)
        c1.metric("Maximum Attained Velocity", f"{opt_v:.2f} m/s", f"{opt_v*3.6:.2f} km/h")
        c2.metric("Optimum Reduction Ratio", f"{opt_g:.2f}")

    else:
        st.error("Excel file must contain 'RPM' and 'Torque' columns.")
else:
    st.info("Please upload an Excel file to see the calculations.")
