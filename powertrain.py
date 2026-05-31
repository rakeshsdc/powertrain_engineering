import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App Title
st.title("Vehicle Power and Traction Limited Velocity Calculator")

# Sidebar for User Inputs (Based on Source Data)
st.sidebar.header("Vehicle Parameters")
m = st.sidebar.slider("Mass (kg)", 100, 1000, 320) # Source: 320kg
r = st.sidebar.slider("Wheel Radius (m)", 0.1, 0.5, 0.291) # Source: 0.291m
crr = st.sidebar.slider("Rolling Resistance Coeff (Crr)", 0.01, 0.1, 0.045) # Source: 0.045
t_max = st.sidebar.slider("Max Engine Torque (Nm)", 10, 200, 85) # Source: 85Nm
rpm_max = st.sidebar.slider("Max Engine RPM", 1000, 8000, 4000) # Source: 4000rpm
eff = st.sidebar.slider("Transmission Efficiency", 0.5, 1.0, 0.9) # Source: 90%
mu = st.sidebar.slider("Coefficient of Friction (mu)", 0.1, 1.2, 0.8)

# Aerodynamic Constants (Assumed for theory)
rho = 1.225 # Air density kg/m3
cd = 0.3    # Drag coefficient
area = 1.5  # Frontal area m2
g = 9.81    # Gravity

# Calculations
# 1. Traction Limit
traction_limit_force = mu * m * g

# 2. Resisitive Forces function
def get_total_resistance(v):
    f_rr = m * g * crr
    f_ad = 0.5 * rho * cd * area * (v**2)
    return f_rr + f_ad

# 3. Velocity and Reduction Ratio Relationship
# Range of Reduction Ratios (G)
g_range = np.linspace(1, 15, 100)
velocities = []
skid_status = []

for G in g_range:
    # Max speed at this G is limited by Max RPM
    v_at_max_rpm = (2 * np.pi * rpm_max * r) / (60 * G)
    
    # Check if Tractive Force at this G exceeds Traction Limit
    tractive_force = (t_max * G * eff) / r
    
    # Power limited check: Can engine power overcome resistance at this speed?
    # Max Power at wheels
    p_max_wheels = (2 * np.pi * rpm_max * t_max * eff) / 60
    
    # Iterate to find terminal velocity where P_avail = P_req
    v_terminal = 0
    for v_test in np.linspace(0.1, v_at_max_rpm, 100):
        power_req = get_total_resistance(v_test) * v_test
        if power_req <= p_max_wheels:
            v_terminal = v_test
        else:
            break
            
    velocities.append(v_terminal)
    skid_status.append(tractive_force > traction_limit_force)

# Visualization
fig, ax = plt.subplots()
ax.plot(g_range, velocities, label="Max Velocity (m/s)")
ax.set_xlabel("Reduction Ratio (G)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Velocity vs Reduction Ratio")
ax.grid(True)

# Highlight Skidding Zone
skid_indices = [i for i, x in enumerate(skid_status) if x]
if skid_indices:
    ax.fill_between(g_range[skid_indices], 0, max(velocities), color='red', alpha=0.2, label="Skid Zone")

ax.legend()
st.pyplot(fig)

# Output Results
max_v = max(velocities)
opt_g = g_range[np.argmax(velocities)]
st.write(f"**Maximum Attainable Velocity:** {max_v:.2f} m/s ({max_v*3.6:.2f} km/h)")
st.write(f"**Optimum Reduction Ratio:** {opt_g:.2f}")

if any(skid_status):
    st.warning("Warning: High reduction ratios in the red zone will cause the vehicle to skid!")
