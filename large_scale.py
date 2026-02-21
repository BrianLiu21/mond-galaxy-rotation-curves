import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

FOLDER_PATH = './data'
FIGURES_PATH = './figures'
KPC_TO_M = 3.086e19
FIXED_ML = 0.5
ALPHA = 4.329e-11
A0_MOND = 1.2e-10
MAX_PLOT_R = 60.0      

def get_alt_grav(v_newtonian, r_m, alpha):
    return np.sqrt((v_newtonian * 1000)**2 + alpha * r_m) / 1000

def get_mond(v_gas, v_disk, v_bulge, ml_ratio, r_m):
    r_m = np.maximum(r_m, 1e-10)
    v_bar_sq = v_gas**2 + ml_ratio * (v_disk**2 + v_bulge**2)
    g_bar = np.maximum(v_bar_sq * 1e6 / r_m, 1e-25)
    
    term = np.exp(-np.sqrt(g_bar / A0_MOND))
    g_mond = g_bar / (1 - term)
    
    return np.sqrt(g_mond * r_m) / 1000

all_radii, all_v_obs, all_v_newt = [], [], []
all_v_gas, all_v_disk, all_v_bulge = [], [], []

data_files = glob.glob(os.path.join(FOLDER_PATH, '*.dat'))

for file_path in data_files:
    df = pd.read_csv(file_path, sep=r'\s+', header=None, comment='#')
    r, v_obs = df[0].values, df[1].values
    v_gas, v_disk, v_bulge = np.abs(df[3].values), np.abs(df[4].values), np.abs(df[5].values)
    mask = (r > 0) & (v_obs > 0)
    if not mask.any(): continue
    r, v_obs = r[mask], v_obs[mask]
    v_gas, v_disk, v_bulge = v_gas[mask], v_disk[mask], v_bulge[mask]
    v_newt = np.sqrt(v_gas**2 + FIXED_ML * (v_disk**2 + v_bulge**2))
    all_radii.append(r); all_v_obs.append(v_obs); all_v_newt.append(v_newt)
    all_v_gas.append(v_gas); all_v_disk.append(v_disk); all_v_bulge.append(v_bulge)


common_r = np.linspace(0.0, MAX_PLOT_R, 1200) 

def interp_avg(curves, radii_list):
    interp_curves = []
    for r, v in zip(radii_list, curves):
        r_ext = np.insert(r, 0, 0.0) if r[0] > 0 else r
        v_ext = np.insert(v, 0, 0.0) if r[0] > 0 else v
        kind = 'cubic' if len(r_ext) > 3 else 'linear'
        f = interp1d(r_ext, v_ext, kind=kind, bounds_error=False, fill_value=np.nan)
        interp_curves.append(f(common_r))
    return np.nanmean(interp_curves, axis=0)

v_obs_smooth = interp_avg(all_v_obs, all_radii)
v_newt_smooth = interp_avg(all_v_newt, all_radii)

alt_grav_list = [get_alt_grav(vn, rd * KPC_TO_M, ALPHA) for rd, vn in zip(all_radii, all_v_newt)]
mond_list = [get_mond(vg, vd, vb, FIXED_ML, rd * KPC_TO_M) 
             for rd, vg, vd, vb in zip(all_radii, all_v_gas, all_v_disk, all_v_bulge)]

v_alt_grav_smooth = interp_avg(alt_grav_list, all_radii)
v_mond_smooth = interp_avg(mond_list, all_radii)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

plt.plot(common_r, v_obs_smooth, color='black', label='Observed (Average)', linewidth=2.5, alpha=0.8)

plt.plot(common_r, v_newt_smooth, color='red', linestyle=':', label='Newtonian (Baryons)', linewidth=2)

plt.plot(common_r, v_mond_smooth, color='#1f77b4', linestyle='-', linewidth=3, 
         label=fr'MOND ($a_0={A0_MOND:.1e}$)')

plt.plot(common_r, v_alt_grav_smooth, color='#2ca02c', linestyle='--', linewidth=3, 
        label=fr'Alternative Gravity Model ($\alpha={ALPHA:.2e}$)')

plt.xlabel('Radius (kpc)', fontsize=14)
plt.ylabel('Rotation Velocity (km/s)', fontsize=14)
plt.title(f'60 kpc Scale Comparison: Models vs. Observed Data', fontsize=16)

plt.xlim(0, MAX_PLOT_R)

valid_max = np.nanmax([
    np.nanmax(v_obs_smooth), 
    np.nanmax(v_alt_grav_smooth), 
    np.nanmax(v_mond_smooth)
])
plt.ylim(0, valid_max * 1.2)

plt.legend(fontsize=12, frameon=True, shadow=True, loc='lower right')
plt.tight_layout()

output_path = os.path.join(FIGURES_PATH, 'LargeScale_Comparison.png')
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Graph saved to: {output_path}")