# 3D Earthquake Simulator

A Python-based 3D Earthquake Simulator designed to analyze the dynamic response of multi-story buildings under seismic excitation.

The project combines structural dynamics, numerical analysis, machine learning, and interactive 3D visualization in a Streamlit dashboard.

## Features

- Multi-degree-of-freedom (MDOF) building model
- Synthetic earthquake ground motion generation
- Structural response analysis using the Newmark-beta method
- Rayleigh damping model
- Natural frequency and mode shape analysis
- Interactive 3D building visualization
- Floor displacement and inter-story drift analysis
- Base shear and absolute acceleration calculation
- PDF and CSV report generation
- Machine learning-based structural response prediction
- Model comparison and feature importance analysis
- Independent validation on unseen scenarios

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
\beta=\frac{1}{4}, \qquad \gamma=\frac{1}{2}
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

The simulator calculates several important structural response parameters:

Inter-Story Drift Ratio

\[
\theta_i(t)=\frac{u_i(t)-u_{i-1}(t)}{h_i}
\]

Absolute Acceleration

\[
\ddot{u}_{abs}(t)=\ddot{u}(t)+a_g(t)
\]

Base Shear

\[
V_b(t)=\sum_{i=1}^{N}m_i\ddot{u}_{i,abs}(t)
\]

## Machine Learning

The project integrates machine learning models to predict structural response parameters using building properties and earthquake ground-motion characteristics.

### Models

Three regression models are implemented and compared:

- Random Forest Regressor
- Extra Trees Regressor
- XGBoost Regressor

### Dataset

The machine learning dataset contains:

- 2,000 synthetic earthquake scenarios
- 12 input features
- 3 prediction targets
- 80/20 training and testing split

Prediction targets:

1. Maximum displacement
2. Maximum inter-story drift ratio
3. Maximum base shear

### Model Evaluation

Models are evaluated using:

- R-squared (R²)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The Extra Trees model achieved the highest mean R² score among the evaluated models.

| Model | Mean R² |
|---|---:|
| Extra Trees | 0.7258 |
| XGBoost | 0.6935 |
| Random Forest | 0.6874 |

### Independent Validation

An additional validation experiment was performed using 50 newly generated scenarios with a separate random seed.

| Prediction Target | R² |
|---|---:|
| Maximum Displacement | 0.670 |
| Maximum Drift Ratio | 0.669 |
| Maximum Base Shear | 0.740 |

This experiment provides an additional assessment of model performance on scenarios not used during training.

## Project Structure

`text
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
├── ml/
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── compare.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── validation.py
│   └── artifacts/
│       ├── best_model.joblib
│       ├── evaluation.json
│       ├── feature_columns.json
│       ├── training_dataset.csv
│       └── test_predictions.csv
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


## Technologies

- Python 3
- NumPy
- SciPy
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Matplotlib
- ReportLab
- Joblib
- Pytest

## Scope and Limitations

The simulator uses a simplified linear shear-building model and synthetic earthquake ground motions.

Machine learning models are trained and evaluated using synthetically generated scenarios. Their predictive performance depends on the quality and range of the training data.

This project is intended for educational purposes and preliminary structural response analysis. It is not a substitute for professional seismic design, real-world structural assessment, or code-compliant engineering software.

## Author

Elham Mavarani