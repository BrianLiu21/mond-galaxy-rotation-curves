import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

FILE_PATH = './data/DDO154_rotmod.dat'  
FIGURES_PATH = './figures'

KPC_TO_M = 3.086e19
FIXED_ML = 0.5
A0_MOND = 1.2e-10

if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

def get_alt_grav(v_newtonian, r_m, alpha):
    v_sq_si = (v_newtonian * 1000)**2 + alpha * r_m
    return np.sqrt(np.maximum(v_sq_si, 0)) / 1000

def get_mond(v_newtonian, r_m):
    r_m = np.maximum(r_m, 1e-10)
    g_bar = (v_newtonian * 1000)**2 / r_m
    g_bar = np.maximum(g_bar, 1e-25) 
    term = np.exp(-np.sqrt(g_bar / A0_MOND))
    g_mond = g_bar / (1 - term)
    return np.sqrt(g_mond * r_m) / 1000

df = pd.read_csv(FILE_PATH, sep=r'\s+', header=None, comment='#')
r_raw = df[0].values
v_obs_raw = df[1].values
v_gas_raw = np.abs(df[3].values)
v_disk_raw = np.abs(df[4].values)
v_bulge_raw = np.abs(df[5].values)
mask = (r_raw > 0) & (v_obs_raw > 0)
r = r_raw[mask]
v_obs = v_obs_raw[mask]
v_gas, v_disk, v_bulge = v_gas_raw[mask], v_disk_raw[mask], v_bulge_raw[mask]
v_newt = np.sqrt(v_gas**2 + FIXED_ML * (v_disk**2 + v_bulge**2))

alpha_grid = np.logspace(-12, -10, 200)
sse_list = [] 
r_m = r * KPC_TO_M

for a in alpha_grid:
    v_model = get_alt_grav(v_newt, r_m, a)
    sse = np.sum((v_obs - v_model)**2)
    sse_list.append(sse)

best_alpha = alpha_grid[np.argmin(sse_list)]
print(f"Optimization Results for {os.path.basename(FILE_PATH)}")
print(f"Best Alpha: {best_alpha:.3e}")

r_smooth = np.linspace(0.1, r.max(), 500)
r_m_smooth = r_smooth * KPC_TO_M

def interp_component(x, y, x_new):
    kind = 'cubic' if len(x) > 4 else 'linear'
    f = interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")
    return np.maximum(f(x_new), 0)

v_gas_s = interp_component(r, v_gas, r_smooth)
v_disk_s = interp_component(r, v_disk, r_smooth)
v_bulge_s = interp_component(r, v_bulge, r_smooth)

v_newt_smooth = np.sqrt(v_gas_s**2 + FIXED_ML * (v_disk_s**2 + v_bulge_s**2))
v_mond_smooth = get_mond(v_newt_smooth, r_m_smooth)
v_alt_grav_smooth = get_alt_grav(v_newt_smooth, r_m_smooth, best_alpha)

v_obs_smooth = interp_component(r, v_obs, r_smooth)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(r_smooth, v_mond_smooth, color='blue', alpha=0.8, linewidth=2, label='MOND (RAR)')
plt.plot(r_smooth, v_alt_grav_smooth, color='green', linestyle='--', linewidth=2,
         label=fr'Alternative Gravity Model ($\alpha$={best_alpha:.2e})')
plt.plot(r_smooth, v_newt_smooth, color='red', linestyle=':', linewidth=1.5, label='Newtonian (Baryons)')

plt.plot(r_smooth, v_obs_smooth, color='black', alpha=0.3, linewidth=1) 
plt.errorbar(r, v_obs, yerr=v_obs*0.05, fmt='o', color='black', ecolor='gray', 
             capsize=3, label='Observed Data') 

plt.xlabel('Radius (kpc)', fontsize=12)
plt.ylabel('Rotation Velocity (km/s)', fontsize=12)
plt.title(f'Rotation Curve Analysis: {os.path.basename(FILE_PATH)}', fontsize=14)
plt.legend(frameon=True, shadow=True, fontsize=11)
plt.xlim(0, r.max() * 1.1)
plt.ylim(0, np.max(v_obs) * 1.3)

output_name = os.path.basename(FILE_PATH).replace('.dat', '_analysis.png')
output_full_path = os.path.join(FIGURES_PATH, output_name)
plt.savefig(output_full_path, dpi=300)
plt.show()

print(f"Plot saved to: {output_full_path}")