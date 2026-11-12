# 3D Earthquake Simulator

A Python-based 3D Earthquake Simulator for analyzing the dynamic response of multi-story buildings under seismic excitation.

The project combines structural dynamics, numerical analysis, and interactive 3D visualization in a Streamlit dashboard.

## Features

- Multi-degree-of-freedom (MDOF) building model
- Synthetic earthquake ground motion generation
- Structural response analysis using the Newmark-beta method
- Rayleigh damping model
- Interactive 3D building visualization
- Natural frequency and mode shape analysis
- Inter-story drift and drift ratio calculation
- Base shear and floor displacement analysis
- PDF and CSV report generation

## Theoretical Background

The building is modeled as a multi-degree-of-freedom shear structure subjected to horizontal ground motion.

### Equation of Motion

\[
\mathbf{M}\ddot{\mathbf{u}}+
\mathbf{C}\dot{\mathbf{u}}+
\mathbf{K}\mathbf{u}
=-\mathbf{M}\mathbf{r}a_g(t)
\]

Where:
- M: Mass matrix
- C: Damping matrix
- K: Stiffness matrix
- u: Relative displacement
- ag(t): Ground acceleration

### Numerical Integration

The structural response is calculated using the Newmark-beta average acceleration method:

\[
\beta=\frac{1}{4},\qquad \gamma=\frac{1}{2}
\]

### Modal Analysis

Natural frequencies and mode shapes are obtained from:

\[
\mathbf{K}\boldsymbol{\phi}
=\omega^2\mathbf{M}\boldsymbol{\phi}
\]

\[
f=\frac{\omega}{2\pi}
\]

### Engineering Metrics

Inter-Story Drift Ratio:

\[
\theta_i(t)=\frac{u_i(t)-u_{i-1}(t)}{h_i}
\]

Absolute Acceleration:

\[
\ddot{u}_{abs}(t)=\ddot{u}(t)+a_g(t)
\]

Base Shear:

\[
V_b(t)=\sum_{i=1}^{N}m_i\ddot{u}_{i,abs}(t)
\]

## Project Structure

3D-Earthquake-Simulator/
│
├── app.py
├── requirements.txt
├── README.md
│
├── physics/
│   ├── building.py
│   ├── earthquake.py
│   ├── solver.py
│   └── engineering_metrics.py
│
├── visualization/
│   ├── plots.py
│   └── building_3d.py
│
├── reports/
│   └── pdf_report.py
│
├── data/
├── assets/
│
├── test_building.py
├── test_earthquake.py
├── test_solver.py
├── test_engineering_metrics.py
├── test_building_3d.py
├── test_plots.py
├── test_solver_validation.py
└── test_engineering_metrics_validation.py
## Installation

Clone the repository:

git clone https://github.com/elhammavarani/3D-Earthquake-Simulator.git
cd 3D-Earthquake-Simulator
Install dependencies:

python -m pip install -r requirements.txt
## Run the Application

python -m streamlit run app.py
The application will open in your browser.

## Run Tests

Install pytest if it is not already installed:

python -m pip install pytest
Run the test suite:

python -m pytest -v
## Technologies

- Python 3
- NumPy
- SciPy
- Streamlit
- Plotly
- Matplotlib
- Pandas

## Scope and Limitations

This project uses a simplified linear shear-building model and synthetic earthquake ground motion. It is intended for educational purposes and preliminary structural response analysis, not as a substitute for professional seismic design or code-compliant engineering software.

## Author

Elham Mavarani

