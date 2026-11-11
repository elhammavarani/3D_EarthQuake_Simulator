from io import BytesIO

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart


def generate_pdf_report(results, metrics, story_height=3.2):
    """
    Generate a PDF engineering report.

    Parameters
    ----------
    results : dict
        Simulation results stored in Streamlit session state.
    metrics : dict
        Output of calculate_engineering_metrics().
    story_height : float
        Height of each story in meters.

    Returns
    -------
    bytes
        Generated PDF file contents.
    """

    # -----------------------------------------------------
    # 1. EXTRACT AND VALIDATE DATA
    # -----------------------------------------------------

    t = np.asarray(results["t"], dtype=float)
    ag = np.asarray(results["ag"], dtype=float)
    u = np.asarray(results["u"], dtype=float)

    n_floors = int(results["n_floors"])

    if (
        t.ndim != 1
        or ag.ndim != 1
        or u.ndim != 2
        or len(t) != len(ag)
        or len(t) != len(u)
        or u.shape[1] != n_floors
        or len(t) < 2
    ):
        raise ValueError("Invalid simulation data for PDF report.")

    if not all(np.all(np.isfinite(x)) for x in (t, ag, u)):
        raise ValueError("Simulation data contains invalid values.")

    if not np.isfinite(story_height) or story_height <= 0:
        raise ValueError("story_height must be positive.")

    buffer = BytesIO()

    # -----------------------------------------------------
    # 2. DOCUMENT SETUP
    # -----------------------------------------------------

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Seismic Structural Analysis Report",
        author="Seismic Lab"
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#12304A"),
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=19,
        textColor=colors.HexColor("#087DB5"),
        spaceBefore=14,
        spaceAfter=9
    ))

    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        textColor=colors.HexColor("#526273")
    ))

    story = []

    # -----------------------------------------------------
    # 3. REPORT HEADER
    # -----------------------------------------------------

    story.append(Paragraph(
        "SEISMIC LAB",
        styles["ReportTitle"]
    ))

    story.append(Paragraph(
        "STRUCTURAL DYNAMIC ANALYSIS REPORT",
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=11,
            textColor=colors.HexColor("#526273"),
            spaceAfter=18
        )
    ))

    story.append(Paragraph(
        "1. Simulation Overview",
        styles["SectionTitle"]
    ))

    overview_data = [
        ["Parameter", "Value"],
        ["Building floors", str(n_floors)],
        ["Story height", f"{story_height:.2f} m"],
        ["Total building height", f"{n_floors * story_height:.2f} m"],
        ["Peak ground acceleration", f'{results["pga_g"]:.3f} g'],
        ["Damping ratio", f'{results["damping_ratio"]:.3f}'],
        ["Earthquake duration", f'{results["duration"]} seconds'],
        ["Time steps", str(len(t))]
    ]

    overview_table = Table(
        overview_data,
        colWidths=[8 * cm, 8 * cm]
    )

    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#243447")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(overview_table)
    story.append(Spacer(1, 12))

    # -----------------------------------------------------
    # 4. ENGINEERING PERFORMANCE INDICATORS
    # -----------------------------------------------------

    story.append(Paragraph(
        "2. Engineering Performance Indicators",
        styles["SectionTitle"]
    ))

    indicators = [
        ["Engineering Parameter", "Result"],
        [
            "Maximum floor displacement",
            f'{np.max(metrics["max_floor_displacement"]):.6f} m'
        ],
        [
            "Maximum story drift",
            f'{np.max(metrics["max_story_drift"]):.6f} m'
        ],
        [
            "Maximum drift ratio",
            f'{np.max(metrics["max_drift_ratio"]) * 100:.4f} %'
        ],
        [
            "Maximum base shear",
            f'{metrics["max_base_shear"] / 1000:.3f} kN'
        ],
        [
            "First natural frequency",
            f'{metrics["first_natural_frequency_hz"]:.4f} Hz'
        ]
    ]

    indicator_table = Table(
        indicators,
        colWidths=[8 * cm, 8 * cm]
    )

    indicator_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#243447")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(indicator_table)

    # -----------------------------------------------------
    # 5. FLOOR-BY-FLOOR RESULTS
    # -----------------------------------------------------

    story.append(Paragraph(
        "3. Floor-by-Floor Response",
        styles["SectionTitle"]
    ))

    floor_data = [[
        "Floor",
        "Max Displacement (m)",
        "Max Drift (m)",
        "Drift Ratio (%)",
        "Mode 1"
    ]]

    for i in range(n_floors):

        floor_data.append([
            str(i + 1),
            f'{metrics["max_floor_displacement"][i]:.5f}',
            f'{metrics["max_story_drift"][i]:.5f}',
            f'{metrics["max_drift_ratio"][i] * 100:.4f}',
            f'{metrics["first_mode_shape"][i]:.4f}'
        ])

    floor_table = Table(
        floor_data,
        colWidths=[
            1.5 * cm,
            3.5 * cm,
            3.0 * cm,
            3.0 * cm,
            2.5 * cm
        ],
        repeatRows=1
    )

    floor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#243447")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(floor_table)

    # -----------------------------------------------------
    # 6. EARTHQUAKE GROUND MOTION CHART
    # -----------------------------------------------------

    story.append(PageBreak())

    story.append(Paragraph(
        "4. Earthquake Ground Motion",
        styles["SectionTitle"]
    ))

    # Downsample for a compact PDF chart.
    sample_indices = np.linspace(
        0,
        len(t) - 1,
        min(250, len(t)),
        dtype=int
    )

    chart = HorizontalLineChart()
    chart.x = 45
    chart.y = 35
    chart.width = 450
    chart.height = 190

    chart_data = [
        [
            float(ag[i])
            for i in sample_indices
        ]
    ]

    chart.data = chart_data
    chart.categoryAxis.categoryNames = [
        f"{t[i]:.1f}"
        for i in sample_indices
    ]

    chart.categoryAxis.labels.fontSize = 6
    chart.categoryAxis.labels.angle = 45
    chart.categoryAxis.labels.boxAnchor = "ne"

    chart.valueAxis.valueMin = float(np.min(ag))
    chart.valueAxis.valueMax = float(np.max(ag))
    chart.valueAxis.labels.fontSize = 8

    chart.lines[0].strokeColor = colors.HexColor("#087DB5")
    chart.lines[0].strokeWidth = 1.5

    chart_drawing = Drawing(500, 250)
    chart_drawing.add(chart)

    story.append(chart_drawing)

    story.append(Paragraph(
        "Ground acceleration is expressed in m/s². "
        "The input motion used in this project is synthetically generated.",
        styles["SmallNote"]
    ))

    # -----------------------------------------------------
    # 7. FLOOR DISPLACEMENT CHART
    # -----------------------------------------------------

    story.append(Paragraph(
        "5. Maximum Floor Displacement",
        styles["SectionTitle"]
    ))

    displacement_chart = VerticalBarChart()
    displacement_chart.x = 45
    displacement_chart.y = 35
    displacement_chart.width = 450
    displacement_chart.height = 190

    displacement_chart.data = [
        [
            float(value)
            for value in metrics["max_floor_displacement"]
        ]
    ]

    displacement_chart.categoryAxis.categoryNames = [
        f"F{i}"
        for i in range(1, n_floors + 1)
    ]

    displacement_chart.categoryAxis.labels.fontSize = 8
    displacement_chart.valueAxis.labels.fontSize = 8

    displacement_chart.bars[0].fillColor = colors.HexColor("#087DB5")
    displacement_chart.valueAxis.valueMin = 0

    displacement_drawing = Drawing(500, 250)
    displacement_drawing.add(displacement_chart)

    story.append(displacement_drawing)

    # -----------------------------------------------------
    # 8. STORY DRIFT CHART
    # -----------------------------------------------------

    story.append(Paragraph(
        "6. Maximum Story Drift",
        styles["SectionTitle"]
    ))

    drift_chart = VerticalBarChart()
    drift_chart.x = 45
    drift_chart.y = 35
    drift_chart.width = 450
    drift_chart.height = 190

    drift_chart.data = [
        [
            float(value)
            for value in metrics["max_story_drift"]
        ]
    ]

    drift_chart.categoryAxis.categoryNames = [
        f"F{i}"
        for i in range(1, n_floors + 1)
    ]

    drift_chart.categoryAxis.labels.fontSize = 8
    drift_chart.valueAxis.labels.fontSize = 8
    drift_chart.bars[0].fillColor = colors.HexColor("#0D9488")
    drift_chart.valueAxis.valueMin = 0

    drift_drawing = Drawing(500, 250)
    drift_drawing.add(drift_chart)

    story.append(drift_drawing)

    # -----------------------------------------------------
    # 9. ENGINEERING DISCLAIMER
    # -----------------------------------------------------

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "Engineering Note",
        styles["SectionTitle"]
    ))

    story.append(Paragraph(
        "This report presents the numerical response of a linear "
        "multi-degree-of-freedom shear building model subjected to "
        "synthetically generated ground acceleration. The reported "
        "quantities are simulation results and do not constitute a "
        "certified structural design, code compliance assessment, "
        "or safety evaluation.",
        styles["SmallNote"]
    ))

    # -----------------------------------------------------
    # 10. BUILD PDF
    # -----------------------------------------------------

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()