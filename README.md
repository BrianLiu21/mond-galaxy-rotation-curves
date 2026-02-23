# MOND Galaxy Rotation Curves Analysis

A comprehensive investigation of galaxy rotation curves using Modified Newtonian Dynamics (MOND) and alternative gravity models.

## Overview

This project compares multiple theoretical frameworks for explaining galaxy rotation curves without invoking dark matter:

- **MOND (Modified Newtonian Dynamics)**: A well-established modified gravity theory that replaces the classical inverse-square law at low accelerations
- **Alternative Gravity Model**: A custom alternative model featuring an alpha parameter correction term

The analysis spans three observational scales to evaluate model performance across different radial distances.

## Project Structure

```
├── single_galaxy.py       # Single galaxy (NGC3198) detailed analysis
├── large_scale.py         # Large radius analysis (up to 60 kpc)
├── small_scale.py         # Small radius analysis (up to 10 kpc)
├── data/                  # Galaxy rotation curve datasets
├── figures/               # Output plots and comparisons
└── README.md              # Project documentation
```

## Analysis Scripts

### single_galaxy.py
Performs detailed analysis on a single galaxy (NGC3198):
- Compares observed rotation curve with MOND and alternative model predictions
- Optimizes the alpha parameter through least-squares fitting
- Generates visualization comparing model predictions to observational data

### large_scale.py
Multi-galaxy analysis at extended radii:
- Tests model performance across multiple galaxies
- Focuses on rotation curves extending to 60 kpc
- Evaluates consistency and parameter variations

### small_scale.py
Focused analysis of inner galaxy regions:
- Concentrates on rotation curves up to 10 kpc
- Fine-tunes parameter optimization for smaller scales
- Investigates model performance in high-curvature regions

## Models

### Modified Newtonian Dynamics (MOND)

$$g_{\text{MOND}} = \frac{g_{\text{bar}}}{1 - e^{-\sqrt{g_{\text{bar}} / a_0}}}$$

Where:
- $g_{\text{bar}}$ = Newtonian gravitational acceleration from baryonic matter
- $a_0 = 1.2 \times 10^{-10}$ m/s² = MOND acceleration scale parameter

### Alternative Gravity Model

$$V_{\text{total}}^2 = V_{\text{Newton}}^2 + \alpha \cdot r$$

Where:
- $V_{\text{Newton}}$ = Classical Newtonian velocity from baryonic mass
- $\alpha$ = Fitting parameter (optimized per galaxy)
- $r$ = Galactocentric radius

## Data Format

The `data/` directory contains rotation curve measurements in ASCII format:
- Column 1: Radial distance (kpc)
- Column 2: Observed rotation velocity (km/s)
- Column 3: Error (km/s)
- Column 4: Gas velocity component (km/s)
- Column 5: Disk velocity component (km/s)
- Column 6: Bulge velocity component (km/s)

## Configuration Parameters

Key parameters configurable in each script:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FIXED_ML` | 0.5 | Mass-to-light ratio |
| `A0_MOND` | 1.2e-10 | MOND acceleration scale (m/s²) |
| `KPC_TO_M` | 3.086e19 | Kiloparsec to meter conversion |

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy

### Setup

```bash
# Clone the repository
git clone https://github.com/BrianLiu21/mond-galaxy-rotation-curves.git
cd mond-galaxy-rotation-curves

# Install dependencies
pip install pandas numpy matplotlib scipy
```

## Usage

Run individual analyses:

```bash
# Analyze a single galaxy with both models
python single_galaxy.py

# Large-scale multi-galaxy analysis
python large_scale.py

# Small-scale multi-galaxy analysis
python small_scale.py
```

Output plots are saved automatically to the `figures/` directory.

## Results

Analysis outputs include:
- Rotation curve comparisons (observed vs. model predictions)
- Best-fit parameter values
- Model residuals and goodness-of-fit metrics
- Multi-galaxy parameter trends

## Author

BrianLiu21

## License

This project is available for academic and research use.
