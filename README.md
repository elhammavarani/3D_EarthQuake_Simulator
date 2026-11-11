# 3D Earthquake Simulator

An interactive structural earthquake response
simulation and engineering analysis platform.

## Features
- Multi-story shear building model
- Synthetic earthquake ground motion
- Newmark-Beta dynamic response solver
- Interactive 3D building animation
- Engineering response metrics
- PDF and CSV report generation

## Technology Stack
- Python
- NumPy
- SciPy
- Streamlit
- Plotly
- Pandas
- ReportLab

## Installation

python -m venv .venv
Activate the environment, then run:

pip install -r requirements.txt
## Run

streamlit run app.py
## Run Tests

pytest -v
## Model Assumptions
- Linear elastic structural behavior
- Lumped floor masses
- Shear building idealization
- Rayleigh damping
- Synthetic earthquake excitation

## Limitations

This project is an educational simulation tool
and is not a certified structural design or
safety assessment application.