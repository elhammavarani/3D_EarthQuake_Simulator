import streamlit as st
import numpy as np
import pandas as pd

from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response

from visualization.plots import (
    plot_ground_motion,
    plot_story_response,
    plot_max_story_response
)

from visualization.building_3d import create_building_3d

from physics.engineering_metrics import calculate_engineering_metrics

from reports.pdf_report import generate_pdf_report

from ml.compare import compare_models

from ml.feature_importance import get_feature_importance


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Seismic Control Center",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ENTERPRISE / MISSION CONTROL THEME
# =========================================================

st.markdown("""
<style>

/* Application background */
.stApp {
    background: #080E19;
    color: #E5EAF2;
}

/* Main content width */
.block-container {
    max-width: 1700px;
    padding: 2rem 2.5rem 3rem 2.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1625;
    border-right: 1px solid #1E3047;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #E5EAF2 !important;
}

/* Headings */
h1, h2, h3 {
    color: #F1F5F9 !important;
    letter-spacing: -0.4px;
}

h1 {
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

/* Text */
p, label, .stMarkdown {
    color: #CBD5E1;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111E30;
    border: 1px solid #243A54;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 115px;
}

[data-testid="stMetricLabel"] {
    color: #91A4BA !important;
    font-size: 0.82rem;
}

[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: #087DB5;
    color: white;
    border: 1px solid #1597D0;
    border-radius: 8px;
    min-height: 42px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #0B91D0;
    border-color: #38BDF8;
    color: white;
}

/* Navigation radio */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #111E30;
    border: 1px solid #243A54;
    border-radius: 8px;
    padding: 10px 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #243A54;
}

.stTabs [data-baseweb="tab"] {
    background: #111E30;
    border-radius: 7px 7px 0 0;
    color: #A8B7C9;
    padding: 10px 16px;
}

.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8;
}

/* Plot containers */
[data-testid="stPlotlyChart"] {
    background: #0D1828;
    border: 1px solid #20334A;
    border-radius: 12px;
    padding: 6px;
}

/* Dividers */
hr {
    border-color: #20334A !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: #111E30;
    border: 1px solid #29415D;
    border-radius: 10px;
}

/* Input controls */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #111E30;
    border-color: #29415D;
}

/* Slider */
[data-testid="stSlider"] {
    padding-top: 8px;
    padding-bottom: 8px;
}

/* Custom panels */
.control-panel {
    background: #101C2D;
    border: 1px solid #263B55;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.eyebrow {
    color: #38BDF8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.muted {
    color: #8FA2B8;
    font-size: 13px;
}

.status-dot {
    color: #34D399;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def render_header(section, description):
    st.markdown(
        f"""
        <div style="
            padding: 8px 0 20px 0;
            border-bottom: 1px solid #20334A;
            margin-bottom: 24px;
        ">
            <div class="eyebrow">SEISMIC CONTROL CENTER / {section}</div>
            <div style="
                font-size: 30px;
                font-weight: 850;
                color: #F1F5F9;
                margin-top: 8px;
            ">
                {section}
            </div>
            <div class="muted" style="margin-top: 5px;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_figure(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1828",
        plot_bgcolor="#0D1828",
        font=dict(
            color="#DCE6F1",
            family="Arial"
        ),
        margin=dict(l=35, r=25, t=55, b=35)
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


def get_results():
    return st.session_state.get("results")


def render_no_results():
    st.info(
        "No simulation results available. "
        "Open the Simulation page and run an analysis first."
    )


def render_kpis(results):
    u = results["u"]
    a = results["a"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "BUILDING FLOORS",
        f'{results["n_floors"]}'
    )

    c2.metric(
        "PEAK GROUND ACCELERATION",
        f'{results["pga_g"]:.2f} g'
    )

    c3.metric(
        "MAX FLOOR DISPLACEMENT",
        f'{np.max(np.abs(u)):.4f} m'
    )

    c4.metric(
        "MAX FLOOR ACCELERATION",
        f'{np.max(np.abs(a)):.3f} m/s²'
    )


# =========================================================
# SIDEBAR / BRANDING / NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding: 8px 4px 22px 4px;
            border-bottom: 1px solid #263B55;
            margin-bottom: 20px;
        ">
            <div style="font-size: 30px;">🏢</div>
            <div style="
                font-size: 18px;
                font-weight: 850;
                color: #F1F5F9;
                margin-top: 6px;
            ">
                SEISMIC LAB
            </div>
            <div style="
                color: #8296AE;
                font-size: 10px;
                letter-spacing: 2px;
            ">
                ENGINEERING CONTROL
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="eyebrow">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Simulation",
            "3D Visualization",
            "Engineering Report",
            "AI Prediction"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="eyebrow">SYSTEM STATUS</div>',
        unsafe_allow_html=True
    )

    if get_results() is None:
        st.markdown(
            '<p class="muted">● Waiting for simulation</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-dot">● Results available</p>',
            unsafe_allow_html=True
        )

    st.divider()

    st.caption("SEISMIC LAB  |  VERSION 1.0")
    st.caption("Structural Dynamics Platform")


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    render_header(
        "Overview",
        "Central monitoring workspace for structural dynamic analysis."
    )

    results = get_results()

    if results is None:
        st.markdown(
            """
            <div class="control-panel">
                <div class="eyebrow">PLATFORM READY</div>
                <h2 style="color:#F1F5F9;">
                    Welcome to Seismic Control Center
                </h2>
                <p style="color:#A8B7C9;">
                    Configure a building model, generate earthquake
                    excitation and evaluate the structural response
                    through interactive engineering visualizations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Start by opening Simulation from the sidebar."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("ANALYSIS ENGINE", "READY")
        c2.metric("3D VISUALIZATION", "READY")
        c3.metric("REPORT MODULE", "READY")

    else:

        render_kpis(results)

        st.divider()

        st.subheader("Latest Analysis")

        col1, col2 = st.columns([1.3, 1])

        with col1:

            st.markdown("#### Earthquake Ground Motion")

            fig = plot_ground_motion(
                results["t"],
                results["ag"]
            )

            st.plotly_chart(
                style_figure(fig, 330),
                use_container_width=True,
                key="overview_ground_motion"
            )

        with col2:

            st.markdown("#### Analysis Configuration")

            st.markdown(
                f"""
                <div class="control-panel">
                    <p class="muted">BUILDING CONFIGURATION</p>
                    <h3>{results["n_floors"]} Floors</h3>
                    <hr>
                    <p class="muted">GROUND MOTION PGA</p>
                    <h3>{results["pga_g"]:.2f} g</h3>
                    <hr>
                    <p class="muted">DAMPING RATIO</p>
                    <h3>{results["damping_ratio"]:.2f}</h3>
                    <hr>
                    <p class="muted">DURATION</p>
                    <h3>{results["duration"]} seconds</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "Ground motion in this project is synthetically generated."
        )


# =========================================================
# PAGE 2: SIMULATION
# =========================================================

elif page == "Simulation":

    render_header(
        "Simulation",
        "Configure the structural model and earthquake excitation."
    )

    st.subheader("01 / Building Configuration")

    col1, col2 = st.columns(2)

    with col1:

        n_floors = st.slider(
            "Number of floors",
            min_value=3,
            max_value=20,
            value=10
        )

    with col2:

        damping_ratio = st.slider(
            "Damping ratio",
            min_value=0.01,
            max_value=0.15,
            value=0.05,
            step=0.01
        )

    st.divider()

    st.subheader("02 / Earthquake Parameters")

    col1, col2 = st.columns(2)

    with col1:

        pga_g = st.slider(
            "Peak Ground Acceleration (g)",
            min_value=0.05,
            max_value=1.0,
            value=0.3,
            step=0.05
        )

    with col2:

        duration = st.slider(
            "Earthquake duration (seconds)",
            min_value=10,
            max_value=60,
            value=30
        )

    st.divider()

    st.subheader("03 / Execute Analysis")

    st.markdown(
        """
        <div class="control-panel">
            <div class="eyebrow">ANALYSIS ENGINE</div>
            <p style="color:#A8B7C9;">
                The system will assemble the structural matrices,
                generate synthetic ground motion and calculate
                displacement, velocity and acceleration using
                the Newmark-beta integration method.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button(
        "▶️  RUN STRUCTURAL SIMULATION",
        type="primary",
        use_container_width=True
    ):

        # Invalidate the previous run before starting a new one.
        # If the new analysis fails, stale results cannot be mistaken
        # for the output of the latest requested simulation.
        st.session_state.pop("results", None)

        try:

            with st.spinner("Running structural dynamic analysis..."):

                M, C, K = create_building(
                    n_floors=n_floors,
                    damping_ratio=damping_ratio
                )

                t, ag = generate_ground_motion(
                    duration=float(duration),
                    pga=pga_g * 9.81
                )

                u, v, a = solve_response(
                    M,
                    C,
                    K,
                    t,
                    ag
                )

                # Calculate engineering metrics once per successful run.
                # Dashboard and reports use the same result snapshot.
                metrics = calculate_engineering_metrics(
                    M=M,
                    K=K,
                    u=u,
                    a=a,
                    ag=ag,
                    story_height=3.2
                )

                st.session_state["results"] = {
                    "t": t,
                    "ag": ag,
                    "u": u,
                    "v": v,
                    "a": a,
                    "M": M,
                    "C": C,
                    "K": K,
                    "metrics": metrics,
                    "story_height": 3.2,
                    "dt": float(t[1] - t[0]),
                    "n_floors": n_floors,
                    "pga_g": pga_g,
                    "duration": duration,
                    "damping_ratio": damping_ratio
                }

            st.success("Structural analysis completed successfully.")

            st.rerun()

        except Exception as error:

            # Ensure a failed run never leaves a partial or stale result.
            st.session_state.pop("results", None)

            st.error(
                f"Simulation failed: {error}"
            )

    results = get_results()

    if results is not None:

        st.divider()
        st.subheader("Latest Simulation Results")
        render_kpis(results)


# =========================================================
# PAGE 3: 3D VISUALIZATION
# =========================================================

elif page == "3D Visualization":

    render_header(
        "3D Visualization",
        "Interactive three-dimensional structural response."
    )

    results = get_results()

    if results is None:

        render_no_results()

    else:

        st.subheader("Building Response Model")

        col1, col2, col3 = st.columns(3)

        with col1:

            amplification = st.slider(
                "Deformation amplification",
                min_value=1,
                max_value=50,
                value=10
            )

        with col2:

            max_frames = st.slider(
                "Animation frames",
                min_value=30,
                max_value=200,
                value=120,
                step=10
            )

        with col3:

            playback_speed = st.slider(
                "Playback speed",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5
            )

        animation_fig = create_building_3d(
            displacement=results["u"],
            time=results["t"],
            time_index=0,
            amplification=amplification,
            max_frames=max_frames,
            playback_speed=playback_speed
        )

        st.plotly_chart(
            animation_fig,
            use_container_width=True,
            key="earthquake_3d_animation"
        )

        st.caption(
            "Deformation amplification is a visual scale, "
            "not the physical displacement."
        )

        st.divider()

        st.subheader("Maximum Floor Displacement")

        max_fig = plot_max_story_response(
            results["u"],
            "Maximum Floor Displacement",
            "Displacement (m)"
        )

        st.plotly_chart(
            style_figure(max_fig, 350),
            use_container_width=True,
            key="3d_max_displacement"
        )


# =========================================================
# PAGE 4: ENGINEERING REPORT
# =========================================================

elif page == "Engineering Report":

    render_header(
        "Engineering Report",
        "Structural response results and analysis plots."
    )

    results = get_results()

    if results is None:

        render_no_results()

    else:

        render_kpis(results)

        st.divider()

        # Use the metrics calculated and stored with this simulation run.
        # This keeps dashboard and report values synchronized.
        metrics = results.get("metrics")

        if metrics is None:
            # Backward compatibility for older session-state data.
            metrics = calculate_engineering_metrics(
                M=results["M"],
                K=results["K"],
                u=results["u"],
                a=results["a"],
                ag=results["ag"],
                story_height=results.get("story_height", 3.2)
            )

            results["metrics"] = metrics

        st.subheader("Engineering Performance Indicators")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Maximum Base Shear",
            f'{metrics["max_base_shear"] / 1000:.2f} kN'
        )

        c2.metric(
            "First Natural Frequency",
            f'{metrics["first_natural_frequency_hz"]:.3f} Hz'
        )

        c3.metric(
            "Maximum Drift Ratio",
            f'{np.max(metrics["max_drift_ratio"]) * 100:.3f} %'
        )

        st.divider()

        st.subheader("Story Drift Analysis")

        drift_df = pd.DataFrame({
            "Floor": [
                f"Floor {i}"
                for i in range(1, results["n_floors"] + 1)
            ],
            "Maximum Displacement (m)": (
                metrics["max_floor_displacement"]
            ),
            "Maximum Story Drift (m)": (
                metrics["max_story_drift"]
            ),
            "Drift Ratio (%)": (
                metrics["max_drift_ratio"] * 100
            )
        })

        st.dataframe(
            drift_df.style.format({
                "Maximum Displacement (m)": "{:.5f}",
                "Maximum Story Drift (m)": "{:.5f}",
                "Drift Ratio (%)": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("First Mode Shape")

        mode_df = pd.DataFrame({
            "Floor": [
                f"Floor {i}"
                for i in range(1, results["n_floors"] + 1)
            ],
            "Normalized Mode Shape": (
                metrics["first_mode_shape"]
            )
        })

        st.bar_chart(
            mode_df.set_index("Floor"),
            y="Normalized Mode Shape"
        )

        st.subheader("Response Analysis")

        tab1, tab2, tab3 = st.tabs([
            "Displacement",
            "Velocity",
            "Acceleration"
        ])

        with tab1:

            fig = plot_story_response(
                results["t"],
                results["u"],
                "Floor Displacement Response",
                "Displacement (m)"
            )

            st.plotly_chart(
                style_figure(fig, 450),
                use_container_width=True,
                key="report_displacement"
            )

            max_fig = plot_max_story_response(
                results["u"],
                "Maximum Floor Displacement",
                "Displacement (m)"
            )

            st.plotly_chart(
                style_figure(max_fig, 380),
                use_container_width=True,
                key="report_max_displacement"
            )

            with tab2:

                    fig = plot_story_response(
                    results["t"],
                    results["v"],
                    "Floor Velocity Response",
                    "Velocity (m/s)"
                )

            st.plotly_chart(
                style_figure(fig, 450),
                use_container_width=True,
                key="report_velocity"
            )

        with tab3:

            fig = plot_story_response(
                results["t"],
                results["a"],
                "Floor Acceleration Response",
                "Acceleration (m/s²)"
            )

            st.plotly_chart(
                style_figure(fig, 450),
                use_container_width=True,
                key="report_acceleration"
            )

        st.divider()

        st.subheader("Earthquake Input")

        ground_fig = plot_ground_motion(
            results["t"],
            results["ag"]
        )

        st.plotly_chart(
            style_figure(ground_fig, 350),
            use_container_width=True,
            key="report_ground_motion"
        )

        # -------------------------------------------------
        # ANALYSIS SUMMARY
        # -------------------------------------------------

        st.subheader("Analysis Summary")

        summary_data = {
            "Parameter": [
                "Number of floors",
                "Peak ground acceleration",
                "Damping ratio",
                "Simulation duration",
                "Maximum displacement",
                "Maximum velocity",
                "Maximum relative acceleration"
            ],
            "Value": [
                str(results["n_floors"]),
                f'{results["pga_g"]:.3f} g',
                f'{results["damping_ratio"]:.3f}',
                f'{results["duration"]} s',
                f'{np.max(np.abs(results["u"])):.6f} m',
                f'{np.max(np.abs(results["v"])):.6f} m/s',
                f'{np.max(np.abs(results["a"])):.6f} m/s²'
            ]
        }

        summary_df = pd.DataFrame(summary_data)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # CSV 1: ANALYSIS SUMMARY
        # -------------------------------------------------

        csv_data = summary_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Analysis Summary (CSV)",
            data=csv_data,
            file_name="earthquake_analysis_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # -------------------------------------------------
        # CSV 2: FLOOR ENGINEERING METRICS
        # -------------------------------------------------

        engineering_csv = drift_df.copy()

        engineering_csv["First Mode Shape"] = (
            metrics["first_mode_shape"]
        )

        engineering_csv_data = engineering_csv.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Floor Engineering Metrics (CSV)",
            data=engineering_csv_data,
            file_name="earthquake_floor_engineering_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

        # -------------------------------------------------
        # CSV 3: COMPLETE TIME-HISTORY RESPONSE
        # -------------------------------------------------

        response_data = {
            "Time (s)": results["t"],
            "Ground Acceleration (m/s²)": results["ag"]
        }

        for floor_index in range(results["n_floors"]):

            floor_number = floor_index + 1

            response_data[
                f"Floor {floor_number} Displacement (m)"
            ] = results["u"][:, floor_index]

            response_data[
                f"Floor {floor_number} Velocity (m/s)"
            ] = results["v"][:, floor_index]

            response_data[
                f"Floor {floor_number} Relative Acceleration (m/s²)"
            ] = results["a"][:, floor_index]

        response_df = pd.DataFrame(response_data)

        response_csv_data = response_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Full Response History (CSV)",
            data=response_csv_data,
            file_name="earthquake_full_response_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        # -------------------------------------------------
        # PDF ENGINEERING REPORT
        # -------------------------------------------------

        pdf_data = generate_pdf_report(
            results=results,
            metrics=metrics,
            story_height=results.get("story_height", 3.2)
        )

        st.download_button(
            label="⬇️ Download Engineering Report (PDF)",
            data=pdf_data,
            file_name="Seismic_Engineering_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.caption(
            "This is a linear structural response simulation "
            "using synthetic earthquake excitation. "
            "It is not a certified structural safety assessment."
        )

        # =========================================================
# PAGE 5: AI PREDICTION
# =========================================================

elif page == "AI Prediction":

    render_header(
        "AI Prediction",
        "Machine-learning-based earthquake response prediction "
        "and comparison with physics simulation."
    )

    st.subheader("01 / Building Configuration")

    col1, col2 = st.columns(2)

    with col1:
        n_floors = st.slider(
            "Number of floors",
            3, 20, 10,
            key="ai_floors"
        )

        mass = st.number_input(
            "Floor mass (kg)",
            min_value=100000,
            max_value=500000,
            value=250000,
            step=10000,
            key="ai_mass"
        )

        story_height = st.number_input(
            "Story height (m)",
            min_value=2.5,
            max_value=5.0,
            value=3.2,
            step=0.1,
            key="ai_story_height"
        )

    with col2:
        stiffness = st.number_input(
            "Story stiffness (N/m)",
            min_value=10_000_000,
            max_value=500_000_000,
            value=150_000_000,
            step=10_000_000,
            key="ai_stiffness"
        )

        damping_ratio = st.slider(
            "Damping ratio",
            0.01, 0.15, 0.05,
            step=0.01,
            key="ai_damping"
        )

    st.divider()

    st.subheader("02 / Earthquake Parameters")

    col1, col2 = st.columns(2)

    with col1:
        pga_g = st.slider(
            "Peak Ground Acceleration (g)",
            0.05, 1.0, 0.3,
            step=0.05,
            key="ai_pga"
        )

        duration = st.slider(
            "Earthquake duration (seconds)",
            10, 60, 15,
            key="ai_duration"
        )

    with col2:
        dt = st.select_slider(
            "Time step (seconds)",
            options=[0.01, 0.02, 0.05],
            value=0.02,
            key="ai_dt"
        )

        seed = st.number_input(
            "Earthquake seed",
            min_value=1,
            max_value=999999,
            value=42,
            key="ai_seed"
        )

    st.divider()

    st.subheader("03 / Run AI Analysis")

    st.markdown(
        """
        <div class="control-panel">
            <div class="eyebrow">HYBRID ANALYSIS ENGINE</div>
            <p style="color:#A8B7C9;">
                Predict structural response using the trained
                Extra Trees model and compare the results against
                the Newmark-beta physics solver.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "RUN AI PREDICTION",
        type="primary",
        use_container_width=True,
        key="run_ai_prediction"
    ):

        try:
            with st.spinner(
                "Running ML prediction and physics simulation..."
            ):

                comparison = compare_models(
                    n_floors=n_floors,
                    mass=mass,
                    stiffness=stiffness,
                    damping_ratio=damping_ratio,
                    story_height=story_height,
                    pga_g=pga_g,
                    duration=float(duration),
                    dt=float(dt),
                    seed=int(seed)
                )

                st.session_state["ai_results"] = comparison

            st.success("AI analysis completed successfully.")

        except Exception as error:
            st.session_state.pop("ai_results", None)
            st.error(f"AI analysis failed: {error}")

    # Display results
    ai_results = st.session_state.get("ai_results")

    if ai_results is not None:

        ml_results = ai_results["ml"]
        physics_results = ai_results["physics"]

        st.divider()

        st.subheader("04 / Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "ML Max Displacement",
            f'{ml_results["max_displacement"]:.4f} m'
        )

        c2.metric(
            "ML Max Drift Ratio",
            f'{ml_results["max_drift_ratio"] * 100:.3f} %'
        )

        c3.metric(
            "ML Max Base Shear",
            f'{ml_results["max_base_shear"] / 1000:.2f} kN'
        )

        st.divider()

        st.subheader("05 / ML vs Physics Comparison")

        comparison_data = []

        labels = {
            "max_displacement": "Max Displacement",
            "max_drift_ratio": "Max Drift Ratio",
            "max_base_shear": "Max Base Shear"
        }

        units = {
            "max_displacement": "m",
            "max_drift_ratio": "%",
            "max_base_shear": "kN"
        }

        for key, label in labels.items():

            ml_value = ml_results[key]
            physics_value = physics_results[key]

            error = (
                abs(ml_value - physics_value)
                / abs(physics_value) * 100
                if physics_value != 0 else 0.0
            )

            comparison_data.append({
                "Parameter": label,
                "ML Prediction": ml_value,
                "Physics Solver": physics_value,
                "Error (%)": error
            })

        comparison_df = pd.DataFrame(comparison_data)

        display_df = comparison_df.copy()

        display_df["ML Prediction"] = [
            f'{row["ML Prediction"] * (100 if key == "max_drift_ratio" else 0.001 if key == "max_base_shear" else 1):.4f} {units[key]}'
            for row, key in zip(comparison_data, labels.keys())
        ]

        display_df["Physics Solver"] = [
            f'{row["Physics Solver"] * (100 if key == "max_drift_ratio" else 0.001 if key == "max_base_shear" else 1):.4f} {units[key]}'
            for row, key in zip(comparison_data, labels.keys())
        ]

        display_df["Error (%)"] = display_df["Error (%)"].map(
            lambda x: f"{x:.2f}%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("06 / Prediction Error")

        st.bar_chart(
            comparison_df.set_index("Parameter")["Error (%)"],
            use_container_width=True
        )

        st.caption(
            "ML predictions are estimates based on synthetic training "
            "scenarios. Physics solver results are numerical reference "
            "results for the same input scenario."
        )

        st.divider()

        st.subheader("07 / Feature Importance")

        importance_df = get_feature_importance()

        importance_df["Importance (%)"] = (
            importance_df["Importance"] * 100
        )

        st.bar_chart(
            importance_df.set_index("Feature")["Importance (%)"],
            use_container_width=True
        )

        st.dataframe(
            importance_df[["Feature", "Importance (%)"]].round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Feature importance represents the relative contribution "
            "of each input feature to the Extra Trees model."
        )

