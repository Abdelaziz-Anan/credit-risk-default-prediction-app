from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "cleaned_credit_risk_data.csv"
MODEL_PATH = APP_DIR / "best_model_xgb.pkl"

REQUIRED_COLUMNS = [
    "Age",
    "Income",
    "Home",
    "Emp_length",
    "Intent",
    "Amount",
    "Rate",
    "High_Interest_Flag",
    "Status",
    "Percent_income",
    "Cred_length",
    "Default",
]

DEFAULT_LABELS = {0: "No Default", 1: "Default"}

st.set_page_config(
    page_title="Credit Risk Analytics and Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_white"


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    return joblib.load(path)


def find_missing_columns(df: pd.DataFrame, expected: Iterable[str]) -> list[str]:
    return [col for col in expected if col not in df.columns]


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_rgb(color: tuple[float, float, float]) -> str:
    return f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"


def _pdf_text(
    commands: list[str],
    x: float,
    y: float,
    text: str,
    font: str = "F1",
    size: int = 11,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    commands.extend(
        [
            "BT",
            f"/{font} {size} Tf",
            f"{_pdf_rgb(color)} rg",
            f"1 0 0 1 {x:.1f} {y:.1f} Tm ({_pdf_escape(str(text))}) Tj",
            "ET",
        ]
    )


def _pdf_fill_rect(
    commands: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[float, float, float],
) -> None:
    commands.extend([f"{_pdf_rgb(color)} rg", f"{x:.1f} {y:.1f} {width:.1f} {height:.1f} re f"])


def _pdf_stroke_rect(
    commands: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[float, float, float],
    line_width: float = 1.0,
) -> None:
    commands.extend(
        [
            f"{line_width:.2f} w",
            f"{_pdf_rgb(color)} RG",
            f"{x:.1f} {y:.1f} {width:.1f} {height:.1f} re S",
        ]
    )


def _pdf_wrap_text(text: str, max_chars: int = 45) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _format_money(value: object) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return str(value)


def _format_number(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def build_scoring_report_pdf(report: dict) -> bytes:
    inputs = report.get("inputs", {})
    generated_raw = str(report.get("generated_at_utc", "-"))
    generated_display = generated_raw
    report_id = "CR-REPORT"
    try:
        parsed_dt = datetime.fromisoformat(generated_raw.replace("Z", "+00:00"))
        generated_display = parsed_dt.strftime("%d %b %Y %H:%M UTC")
        report_id = f"CR-{parsed_dt.strftime('%Y%m%d-%H%M%S')}"
    except Exception:
        pass

    prediction = str(report.get("prediction", "-"))
    default_probability = str(report.get("default_probability", "-"))
    probability_value = float(report.get("default_probability_value", 0.0))
    probability_value = min(max(probability_value, 0.0), 1.0)
    risk_band = str(report.get("risk_band", "-"))

    risk_colors = {
        "High": (0.70, 0.12, 0.12),
        "Medium": (0.77, 0.47, 0.06),
        "Low": (0.09, 0.47, 0.21),
    }
    risk_color = risk_colors.get(risk_band, (0.22, 0.24, 0.29))

    status_value = int(inputs.get("Status", 0)) if str(inputs.get("Status", "0")).isdigit() else 0
    previous_status = "Previously Paid" if status_value == 1 else "Previously Not Paid"
    high_interest = "Yes" if int(inputs.get("High_Interest_Flag", 0)) == 1 else "No"
    recommendation = {
        "High": "Recommendation: Manual approval only with strict credit controls.",
        "Medium": "Recommendation: Route to analyst review before final decision.",
        "Low": "Recommendation: Proceed with standard underwriting checks.",
    }.get(risk_band, "Recommendation: Follow standard risk governance process.")

    dark_blue = (0.07, 0.15, 0.30)
    slate = (0.16, 0.18, 0.22)
    light_bg = (0.96, 0.97, 0.99)
    border = (0.85, 0.88, 0.92)
    white = (1.0, 1.0, 1.0)

    commands: list[str] = []

    _pdf_fill_rect(commands, 0, 0, 595, 842, white)

    _pdf_fill_rect(commands, 0, 758, 595, 84, dark_blue)
    _pdf_text(commands, 36, 806, "Credit Risk Applicant Scoring Report", font="F2", size=18, color=white)
    _pdf_text(commands, 36, 786, "Decision Support Summary", font="F1", size=11, color=(0.85, 0.90, 0.98))
    _pdf_text(commands, 388, 806, f"Report ID: {report_id}", font="F1", size=10, color=white)
    _pdf_text(commands, 388, 788, f"Generated: {generated_display}", font="F1", size=10, color=(0.85, 0.90, 0.98))

    metric_y = 654
    metric_w = 167
    metric_h = 84
    metric_gap = 14
    start_x = 36
    metric_cards = [
        {
            "label": "Predicted Outcome",
            "value": prediction,
            "background": light_bg,
            "label_color": (0.32, 0.36, 0.42),
            "value_color": slate,
            "note": "",
        },
        {
            "label": "Default Probability",
            "value": default_probability,
            "background": light_bg,
            "label_color": (0.32, 0.36, 0.42),
            "value_color": slate,
            "note": "",
        },
        {
            "label": "Risk Band",
            "value": risk_band.upper(),
            "background": risk_color,
            "label_color": white,
            "value_color": white,
            "note": f"Band from probability: {default_probability}",
        },
    ]

    for idx, card in enumerate(metric_cards):
        x = start_x + idx * (metric_w + metric_gap)
        border_color = white if card["label"] == "Risk Band" else border
        _pdf_fill_rect(commands, x, metric_y, metric_w, metric_h, card["background"])
        _pdf_stroke_rect(commands, x, metric_y, metric_w, metric_h, border_color, line_width=1.2)
        _pdf_text(commands, x + 12, metric_y + 58, card["label"], font="F2", size=11, color=card["label_color"])
        _pdf_text(commands, x + 12, metric_y + 28, card["value"], font="F2", size=18, color=card["value_color"])
        if card["note"]:
            _pdf_text(
                commands,
                x + 12,
                metric_y + 12,
                card["note"],
                font="F1",
                size=8,
                color=(0.92, 0.97, 0.94),
            )

    _pdf_text(
        commands,
        36,
        642,
        "Risk Band Rule: Low < 25% | Medium 25% - 49.99% | High >= 50%",
        font="F2",
        size=10,
        color=slate,
    )

    left_x = 36
    right_x = 304
    panel_y = 416
    panel_w = 255
    panel_h = 214

    _pdf_fill_rect(commands, left_x, panel_y, panel_w, panel_h, white)
    _pdf_stroke_rect(commands, left_x, panel_y, panel_w, panel_h, border)
    _pdf_fill_rect(commands, left_x, panel_y + panel_h - 28, panel_w, 28, light_bg)
    _pdf_text(commands, left_x + 10, panel_y + panel_h - 18, "Applicant Profile", font="F2", size=11, color=slate)

    left_lines = [
        f"Age: {_format_number(inputs.get('Age', '-'), 0)} years",
        f"Annual Income: {_format_money(inputs.get('Income', '-'))}",
        f"Home Ownership: {inputs.get('Home', '-')}",
        f"Employment Length: {_format_number(inputs.get('Emp_length', '-'), 0)} years",
        f"Loan Purpose: {inputs.get('Intent', '-')}",
    ]
    y = panel_y + panel_h - 48
    for line in left_lines:
        wrapped = _pdf_wrap_text(line, max_chars=34)
        for w_line in wrapped:
            _pdf_text(commands, left_x + 12, y, w_line, font="F1", size=10, color=slate)
            y -= 16
        y -= 2

    _pdf_fill_rect(commands, right_x, panel_y, panel_w, panel_h, white)
    _pdf_stroke_rect(commands, right_x, panel_y, panel_w, panel_h, border)
    _pdf_fill_rect(commands, right_x, panel_y + panel_h - 28, panel_w, 28, light_bg)
    _pdf_text(commands, right_x + 10, panel_y + panel_h - 18, "Loan and Affordability", font="F2", size=11, color=slate)

    right_lines = [
        f"Requested Loan Amount: {_format_money(inputs.get('Amount', '-'))}",
        f"Interest Rate: {_format_number(inputs.get('Rate', '-'), 2)}%",
        f"Loan to Income: {_format_number(float(inputs.get('Percent_income', 0.0)) * 100, 2)}%",
        f"High Interest Segment: {high_interest}",
        f"Previous Loan Status: {previous_status}",
    ]
    y = panel_y + panel_h - 48
    for line in right_lines:
        wrapped = _pdf_wrap_text(line, max_chars=35)
        for w_line in wrapped:
            _pdf_text(commands, right_x + 12, y, w_line, font="F1", size=10, color=slate)
            y -= 16
        y -= 2

    gauge_x, gauge_y, gauge_w, gauge_h = 36, 274, 523, 122
    _pdf_fill_rect(commands, gauge_x, gauge_y, gauge_w, gauge_h, white)
    _pdf_stroke_rect(commands, gauge_x, gauge_y, gauge_w, gauge_h, border)
    _pdf_fill_rect(commands, gauge_x, gauge_y + gauge_h - 28, gauge_w, 28, light_bg)
    _pdf_text(commands, gauge_x + 10, gauge_y + gauge_h - 18, "Risk Probability Indicator", font="F2", size=11, color=slate)
    _pdf_text(
        commands,
        gauge_x + 12,
        gauge_y + 72,
        f"Default Probability: {default_probability}",
        font="F2",
        size=12,
        color=slate,
    )

    bar_x, bar_y, bar_w, bar_h = gauge_x + 12, gauge_y + 40, gauge_w - 24, 16
    _pdf_fill_rect(commands, bar_x, bar_y, bar_w, bar_h, (0.90, 0.92, 0.95))
    _pdf_fill_rect(commands, bar_x, bar_y, bar_w * probability_value, bar_h, risk_color)
    _pdf_stroke_rect(commands, bar_x, bar_y, bar_w, bar_h, border, line_width=0.8)
    _pdf_text(commands, bar_x, bar_y - 16, "0%", font="F1", size=9, color=(0.40, 0.45, 0.52))
    _pdf_text(commands, bar_x + bar_w - 18, bar_y - 16, "100%", font="F1", size=9, color=(0.40, 0.45, 0.52))

    rec_lines = _pdf_wrap_text(recommendation, max_chars=80)
    rec_y = gauge_y + 16
    for rec_line in rec_lines:
        _pdf_text(commands, gauge_x + 12, rec_y, rec_line, font="F1", size=10, color=slate)
        rec_y -= 14

    _pdf_text(
        commands,
        36,
        44,
        "Confidential - Generated by Credit Risk Analytics Application",
        font="F1",
        size=9,
        color=(0.35, 0.39, 0.45),
    )
    _pdf_text(
        commands,
        411,
        44,
        "Page 1 of 1",
        font="F1",
        size=9,
        color=(0.35, 0.39, 0.45),
    )

    content_stream = "\n".join(commands).encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 5 0 R /Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
        + content_stream
        + b"\nendstream\nendobj\n",
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n",
    ]

    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(obj)

    xref_start = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_bytes.extend(f"{offset:010} 00000 n \n".encode("ascii"))

    pdf_bytes.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("ascii")
    )
    return bytes(pdf_bytes)


def get_categorical_options(model, feature: str, fallback: list[str]) -> list[str]:
    try:
        preprocessor = model.named_steps["preprocessor"]
        for _, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, "categories_") and feature in columns:
                idx = list(columns).index(feature)
                return [str(item) for item in transformer.categories_[idx]]
    except Exception:
        pass
    return sorted(str(item) for item in fallback)


def render_header(df: pd.DataFrame, show_kpis: bool = False) -> None:
    st.title("Credit Risk Analytics and Default Prediction")

    if not show_kpis:
        return

    total_records = len(df)
    default_rate = df["Default"].mean()
    avg_income = df["Income"].mean()
    avg_amount = df["Amount"].mean()
    high_interest_share = df["High_Interest_Flag"].mean()
    card_values = [
        ("Applications", f"{total_records:,}", "#1d4ed8"),
        ("Default Rate", format_percent(default_rate), "#b91c1c"),
        ("Avg Income", f"${avg_income:,.0f}", "#047857"),
        ("Avg Loan Amount", f"${avg_amount:,.0f}", "#7c3aed"),
        ("High Interest Share", format_percent(high_interest_share), "#b45309"),
    ]

    for col, (label, value, color) in zip(st.columns(5), card_values):
        col.markdown(
            f"""
            <div style="background:{color};padding:12px;border-radius:10px;">
                <div style="color:#f8fafc;font-size:0.85rem;">{label}</div>
                <div style="color:white;font-size:1.35rem;font-weight:700;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_key_insights(df: pd.DataFrame) -> None:
    intent_profile = df.groupby("Intent").agg(applications=("Default", "size"), default_rate=("Default", "mean"))
    intent_profile["default_cases"] = intent_profile["applications"] * intent_profile["default_rate"]

    by_home = df.groupby("Home").agg(applications=("Default", "size"), default_rate=("Default", "mean"))
    by_home["default_cases"] = by_home["applications"] * by_home["default_rate"]

    high_interest_default = df[df["High_Interest_Flag"] == 1]["Default"].mean()
    normal_interest_default = df[df["High_Interest_Flag"] == 0]["Default"].mean()

    top_intent_by_rate = intent_profile["default_rate"].idxmax()
    top_intent_by_cases = intent_profile["default_cases"].idxmax()
    top_home_by_rate = by_home["default_rate"].idxmax()
    top_home_by_cases = by_home["default_cases"].idxmax()

    st.markdown("### Key Insights")
    st.markdown(
        f"""
        <div style="background:#0f172a;padding:14px;border-radius:10px;">
            <ul style="color:#e2e8f0;margin:0;">
                <li>Highest-risk intent by rate: <b>{top_intent_by_rate}</b> at <b>{format_percent(intent_profile.loc[top_intent_by_rate, "default_rate"])}</b>.</li>
                <li>Largest intent impact on target (default cases): <b>{top_intent_by_cases}</b> with around <b>{intent_profile.loc[top_intent_by_cases, "default_cases"]:.0f}</b> defaults.</li>
                <li>Highest-risk home segment by rate: <b>{top_home_by_rate}</b> at <b>{format_percent(by_home.loc[top_home_by_rate, "default_rate"])}</b>.</li>
                <li>Largest home-segment impact on target: <b>{top_home_by_cases}</b> with around <b>{by_home.loc[top_home_by_cases, "default_cases"]:.0f}</b> defaults.</li>
                <li>High-interest loans default at <b>{format_percent(high_interest_default)}</b> vs <b>{format_percent(normal_interest_default)}</b> for normal-interest loans.</li>
                <li>Portfolio baseline default rate: <b>{format_percent(df["Default"].mean())}</b>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_executive_summary(df: pd.DataFrame) -> None:
    render_header(df, show_kpis=True)

    summary_tab, segments_tab = st.tabs(["Executive View", "Risk Segments"])

    with summary_tab:
        labeled_df = df.copy()
        labeled_df["Default_Label"] = labeled_df["Default"].map(DEFAULT_LABELS)
        intent_metrics = (
            labeled_df.groupby("Intent", as_index=False)
            .agg(applications=("Default", "size"), default_rate=("Default", "mean"))
            .sort_values("default_rate", ascending=False)
        )
        intent_metrics["default_cases"] = (
            intent_metrics["applications"] * intent_metrics["default_rate"]
        )

        default_share = (
            labeled_df["Default_Label"].value_counts().rename_axis("Default_Label").reset_index(name="Count")
        )
        by_home = (
            labeled_df.groupby("Home", as_index=False)
            .agg(applications=("Default", "size"), default_rate=("Default", "mean"))
            .sort_values("default_rate", ascending=False)
        )
        by_home["default_cases"] = by_home["applications"] * by_home["default_rate"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                intent_metrics,
                x="Intent",
                y="default_rate",
                color="default_rate",
                color_continuous_scale="Reds",
                labels={"default_rate": "Default Rate"},
                title="Default Rate by Loan Purpose",
            )
            fig.update_layout(xaxis_title="Loan Purpose", yaxis_title="Default Rate")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                intent_metrics,
                x="applications",
                y="default_rate",
                size="default_cases",
                color="default_cases",
                color_continuous_scale="Reds",
                labels={
                    "applications": "Applications",
                    "default_rate": "Default Rate",
                    "default_cases": "Expected Defaults",
                },
                title="Intent Risk Impact Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                by_home,
                x="Home",
                y="default_rate",
                color="default_rate",
                color_continuous_scale="Reds",
                labels={"default_rate": "Default Rate"},
                title="Default Rate by Home Ownership",
            )
            fig.update_layout(xaxis_title="Home Ownership", yaxis_title="Default Rate")
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = px.box(
                labeled_df,
                x="Default_Label",
                y="Rate",
                color="Default_Label",
                labels={"Default_Label": "Outcome", "Rate": "Interest Rate"},
                title="Interest Rate Distribution by Outcome",
            )
            st.plotly_chart(fig, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            fig = px.pie(
                default_share,
                names="Default_Label",
                values="Count",
                title="Portfolio Default vs Non-Default Mix",
                hole=0.45,
                color="Default_Label",
                color_discrete_map={"Default": "#b91c1c", "No Default": "#15803d"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with c6:
            impact_table = intent_metrics[["Intent", "applications", "default_rate", "default_cases"]].copy()
            impact_table["default_rate"] = impact_table["default_rate"].map(format_percent)
            impact_table["default_cases"] = impact_table["default_cases"].round(0).astype(int)
            st.subheader("Top Intent Segments by Target Impact")
            st.dataframe(impact_table.head(6), use_container_width=True, hide_index=True)

        render_key_insights(df)

    with segments_tab:
        by_intent = (
            df.groupby("Intent", as_index=False)
            .agg(
                applications=("Default", "size"),
                default_rate=("Default", "mean"),
                avg_amount=("Amount", "mean"),
            )
            .sort_values("default_rate", ascending=False)
        )
        by_intent["default_cases"] = by_intent["applications"] * by_intent["default_rate"]
        by_intent["target_impact_share"] = by_intent["default_cases"] / by_intent["default_cases"].sum()

        by_home = (
            df.groupby("Home", as_index=False)
            .agg(
                applications=("Default", "size"),
                default_rate=("Default", "mean"),
                avg_income=("Income", "mean"),
            )
            .sort_values("default_rate", ascending=False)
        )
        by_home["default_cases"] = by_home["applications"] * by_home["default_rate"]
        by_home["target_impact_share"] = by_home["default_cases"] / by_home["default_cases"].sum()

        by_intent_display = by_intent.copy()
        by_intent_display["default_rate"] = by_intent_display["default_rate"].map(format_percent)
        by_intent_display["avg_amount"] = by_intent_display["avg_amount"].map(lambda x: f"${x:,.0f}")
        by_intent_display["default_cases"] = by_intent_display["default_cases"].round(0).astype(int)
        by_intent_display["target_impact_share"] = by_intent_display["target_impact_share"].map(format_percent)

        by_home_display = by_home.copy()
        by_home_display["default_rate"] = by_home_display["default_rate"].map(format_percent)
        by_home_display["avg_income"] = by_home_display["avg_income"].map(lambda x: f"${x:,.0f}")
        by_home_display["default_cases"] = by_home_display["default_cases"].round(0).astype(int)
        by_home_display["target_impact_share"] = by_home_display["target_impact_share"].map(format_percent)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk by Loan Purpose")
            st.dataframe(by_intent_display, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Risk by Home Ownership")
            st.dataframe(by_home_display, use_container_width=True, hide_index=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                by_intent,
                x="Intent",
                y="default_cases",
                color="default_rate",
                color_continuous_scale="Reds",
                labels={"default_cases": "Expected Defaults", "default_rate": "Default Rate"},
                title="Default Cases Contribution by Loan Purpose",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.bar(
                by_home,
                x="Home",
                y="default_cases",
                color="default_rate",
                color_continuous_scale="Reds",
                labels={"default_cases": "Expected Defaults", "default_rate": "Default Rate"},
                title="Default Cases Contribution by Home Segment",
            )
            st.plotly_chart(fig, use_container_width=True)



def render_portfolio_analysis(df: pd.DataFrame) -> None:
    render_header(df, show_kpis=False)
    st.subheader("Portfolio Analysis")
    st.caption("Detailed exploration across univariate, bivariate, and multivariate perspectives.")

    st.sidebar.markdown("### Portfolio Filters")
    home_options = sorted(df["Home"].dropna().unique().tolist())
    intent_options = sorted(df["Intent"].dropna().unique().tolist())

    selected_home = st.sidebar.multiselect("Home Ownership", home_options, default=home_options)
    selected_intent = st.sidebar.multiselect("Loan Purpose", intent_options, default=intent_options)

    min_rate = float(df["Rate"].min())
    max_rate = float(df["Rate"].max())
    selected_rate = st.sidebar.slider(
        "Interest Rate Range",
        min_value=float(round(min_rate, 2)),
        max_value=float(round(max_rate, 2)),
        value=(float(round(min_rate, 2)), float(round(max_rate, 2))),
    )

    filtered_df = df[
        df["Home"].isin(selected_home)
        & df["Intent"].isin(selected_intent)
        & df["Rate"].between(selected_rate[0], selected_rate[1])
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Adjust the filter values.")
        return

    filtered_df = filtered_df.copy()
    filtered_df["Default_Label"] = filtered_df["Default"].map(DEFAULT_LABELS)
    high_interest_threshold = 12.0

    uni_tab, bi_tab, multi_tab = st.tabs(
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    with uni_tab:
        st.markdown("### Distribution of Applicant Age")
        fig_age = px.histogram(filtered_df, x="Age", nbins=30, color_discrete_sequence=["#1d4ed8"])
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown(
            "**Insight:** Most applicants are concentrated in the young-to-mid career range, which is the bank's largest borrower base."
        )
        st.markdown("---")

        st.markdown("### Distribution of Applicant Income")
        fig_income = px.violin(
            filtered_df,
            x="Default_Label",
            y="Income",
            box=True,
            color="Default_Label",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig_income, use_container_width=True)
        st.markdown(
            "**Insight:** Income distribution is right-skewed with clear outliers; most decisions should focus on the core income band."
        )
        st.markdown("---")

        st.markdown("### Home Ownership Distribution")
        fig_home = px.pie(
            filtered_df,
            names="Home",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.35,
        )
        st.plotly_chart(fig_home, use_container_width=True)
        st.markdown(
            "**Insight:** Rent and mortgage segments dominate portfolio volume, so small policy changes in these groups have large impact."
        )
        st.markdown("---")

        st.markdown("### Employment Length Distribution")
        fig_emp = px.histogram(
            filtered_df,
            x="Emp_length",
            nbins=25,
            color_discrete_sequence=["#047857"],
        )
        st.plotly_chart(fig_emp, use_container_width=True)
        st.markdown(
            "**Insight:** Most applicants have limited-to-mid tenure, which supports conservative assumptions for early career stability."
        )
        st.markdown("---")

        st.markdown("### Loan Purpose Distribution")
        intent_counts = filtered_df["Intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig_intent = px.bar(
            intent_counts,
            x="Intent",
            y="Count",
            color="Intent",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_intent.update_layout(showlegend=False, xaxis_title="Loan Purpose", yaxis_title="Applications")
        st.plotly_chart(fig_intent, use_container_width=True)
        st.markdown(
            "**Insight:** Portfolio demand is concentrated in a few purposes, creating clear priorities for risk segmentation and pricing."
        )
        st.markdown("---")

        st.markdown("### Distribution of Interest Rates")
        fig_rate = px.histogram(filtered_df, x="Rate", nbins=30, color_discrete_sequence=["#7c3aed"])
        st.plotly_chart(fig_rate, use_container_width=True)
        st.markdown(
            "**Insight:** Most loans are issued in a middle-rate range, while the high-rate tail should be monitored for risk concentration."
        )
        st.markdown("---")

        st.markdown("### High vs Normal Interest Proportion")
        flag_counts = (
            filtered_df["High_Interest_Flag"]
            .value_counts()
            .rename(index={1: "High Interest", 0: "Normal Interest"})
            .reset_index()
        )
        flag_counts.columns = ["Interest Type", "Count"]
        fig_flag = px.pie(
            flag_counts,
            names="Interest Type",
            values="Count",
            hole=0.4,
            color_discrete_map={"High Interest": "#b91c1c", "Normal Interest": "#15803d"},
        )
        st.plotly_chart(fig_flag, use_container_width=True)
        st.markdown(
            f"**Insight:** High-interest loans are defined as rate > {high_interest_threshold:.0f}%, and this segment is central in risk targeting."
        )
        st.markdown("---")

        st.markdown("### Loan Amount as Percentage of Income")
        fig_percent = px.histogram(
            filtered_df,
            x="Percent_income",
            nbins=30,
            color_discrete_sequence=["#b45309"],
        )
        st.plotly_chart(fig_percent, use_container_width=True)
        st.markdown(
            "**Insight:** As loan-to-income ratio rises, affordability pressure increases, which is usually linked to higher default risk."
        )
        st.markdown("---")

        st.markdown("### Target Distribution (Default)")
        target_counts = (
            filtered_df["Default_Label"].value_counts().rename_axis("Default_Label").reset_index(name="Count")
        )
        fig_default = px.pie(
            target_counts,
            names="Default_Label",
            values="Count",
            color="Default_Label",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
            hole=0.4,
        )
        st.plotly_chart(fig_default, use_container_width=True)
        st.markdown(
            "**Insight:** Target class distribution confirms that defaults are meaningful and large enough to support segment-level risk strategy."
        )
        st.markdown("---")

        st.markdown("### Credit History Length")
        fig_cred = px.box(
            filtered_df,
            x="Default_Label",
            y="Cred_length",
            color="Default_Label",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
            points="outliers",
        )
        st.plotly_chart(fig_cred, use_container_width=True)
        st.markdown(
            "**Insight:** Credit-history profile differences can support underwriting cutoffs when used with affordability and pricing signals."
        )
        st.markdown("---")

    with bi_tab:
        st.markdown("### Age vs Default")
        fig = px.violin(
            filtered_df,
            x="Default_Label",
            y="Age",
            color="Default_Label",
            box=True,
            points="all",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Age alone shows limited separation power for target prediction.")
        st.markdown("---")

        st.markdown("### Income vs Default")
        income_cap = filtered_df["Income"].quantile(0.99)
        fig = px.histogram(
            filtered_df[filtered_df["Income"] <= income_cap],
            x="Income",
            color="Default_Label",
            nbins=40,
            barmode="overlay",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Income contributes, but not as strongly as affordability and pricing variables.")
        st.markdown("---")

        st.markdown("### Loan-to-Income Ratio vs Default")
        fig = px.box(
            filtered_df,
            x="Default_Label",
            y="Percent_income",
            color="Default_Label",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Higher loan-to-income ratio is one of the clearest signals linked with default.")
        st.markdown("---")

        st.markdown("### Loan Purpose vs Default")
        intent_default = filtered_df.groupby("Intent", as_index=False)["Default"].mean()
        intent_default["Default_Rate (%)"] = intent_default["Default"] * 100
        fig = px.bar(
            intent_default,
            x="Intent",
            y="Default_Rate (%)",
            color="Default_Rate (%)",
            color_continuous_scale="Reds",
            title="Default Rate by Loan Purpose",
        )
        fig.update_traces(
            text=intent_default["Default_Rate (%)"].round(1).astype(str) + "%",
            textposition="outside",
        )
        fig.update_layout(yaxis_title="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Loan purpose is valuable for segmentation and can drive policy-level risk controls.")
        st.markdown("---")

        st.markdown("### Interest Rate vs Default")
        fig = px.violin(
            filtered_df,
            x="Default_Label",
            y="Rate",
            color="Default_Label",
            box=True,
            points="outliers",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Interest-rate distribution differs by outcome, supporting pricing-aware risk assessment.")
        st.markdown("---")

        st.markdown("### High Interest Segment vs Default")
        flag_default = filtered_df.groupby("High_Interest_Flag", as_index=False)["Default"].mean()
        flag_default["Interest_Level"] = flag_default["High_Interest_Flag"].map(
            {1: "High Interest", 0: "Normal Interest"}
        )
        flag_default["Default_Rate (%)"] = flag_default["Default"] * 100
        fig = px.bar(
            flag_default,
            x="Interest_Level",
            y="Default_Rate (%)",
            color="Default_Rate (%)",
            color_continuous_scale="Reds",
        )
        fig.update_traces(
            text=flag_default["Default_Rate (%)"].round(1).astype(str) + "%",
            textposition="outside",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** High-interest segment has materially higher default rate, making it a primary target lever."
        )
        st.markdown("---")

        st.markdown("### Home Ownership vs Default")
        home_default = filtered_df.groupby("Home", as_index=False)["Default"].mean()
        home_default["Default_Rate (%)"] = home_default["Default"] * 100
        fig = px.bar(
            home_default,
            x="Home",
            y="Default_Rate (%)",
            color="Default_Rate (%)",
            color_continuous_scale="Reds",
        )
        fig.update_traces(
            text=home_default["Default_Rate (%)"].round(1).astype(str) + "%",
            textposition="outside",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** Home ownership type differentiates target risk and is useful for segment-based pricing and limits."
        )
        st.markdown("---")

        st.markdown("### Employment Length vs Default")
        fig = px.scatter(
            filtered_df,
            x="Emp_length",
            y="Percent_income",
            color="Default_Label",
            opacity=0.55,
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
            labels={"Emp_length": "Employment Length", "Percent_income": "Loan as % of Income"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** Affordability stress remains visible across employment tenures, with stronger concentration among defaulters."
        )
        st.markdown("---")

        st.markdown("### Loan Amount vs Default")
        fig = px.histogram(
            filtered_df,
            x="Amount",
            color="Default_Label",
            nbins=40,
            barmode="overlay",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** Larger loan sizes require tighter control when combined with high rate and high loan-to-income ratio."
        )
        st.markdown("---")

    with multi_tab:
        st.markdown("### Correlation Heatmap")
        numeric_cols = [
            "Age",
            "Income",
            "Emp_length",
            "Amount",
            "Rate",
            "High_Interest_Flag",
            "Status",
            "Percent_income",
            "Cred_length",
            "Default",
        ]
        corr = filtered_df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr.round(2),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** Affordability and pricing features show the strongest relation to target compared with demographic variables."
        )
        st.markdown("---")

        st.markdown("### Interest Rate vs Default by Home Ownership")
        fig = px.box(
            filtered_df,
            x="Home",
            y="Rate",
            color="Default_Label",
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** Rate and home segment interaction helps identify pockets where pricing and borrower profile jointly elevate risk."
        )
        st.markdown("---")

        st.markdown("### Loan-to-Income vs Default by Loan Purpose")
        fig = px.violin(
            filtered_df,
            x="Intent",
            y="Percent_income",
            color="Default_Label",
            box=True,
            color_discrete_map={"No Default": "#15803d", "Default": "#b91c1c"},
            labels={"Percent_income": "Loan as % of Income"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Insight:** Across purposes, defaulters trend toward higher affordability stress, confirming loan-to-income as a core risk driver."
        )
        st.markdown("---")



def render_scoring(df: pd.DataFrame, model) -> None:
    render_header(df, show_kpis=False)
    st.subheader("Applicant Scoring")

    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        st.error("The loaded model does not expose input feature names.")
        return

    home_options = get_categorical_options(model, "Home", df["Home"].dropna().unique().tolist())
    intent_options = get_categorical_options(model, "Intent", df["Intent"].dropna().unique().tolist())

    with st.form("scoring_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income", min_value=1, value=45000, step=1000)
            home = st.selectbox("Home Ownership", options=home_options)
            emp_length = st.slider("Employment Length (Years)", min_value=0, max_value=40, value=5)
            intent = st.selectbox("Loan Purpose", options=intent_options)

        with c2:
            amount = st.number_input("Requested Loan Amount", min_value=500, value=10000, step=500)
            rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)
            status_text = st.selectbox(
                "Previous Loan Status",
                options=["1 - Previously Paid", "0 - Previously Not Paid"],
            )
            status = int(status_text.split(" ")[0])

        submitted = st.form_submit_button("Score Applicant", type="primary")

    percent_income = amount / income
    high_interest_flag = int(rate > 12.0)

    st.caption(f"Derived Feature - Percent_income: {percent_income:.2%}")
    st.caption(f"Derived Feature - High_Interest_Flag: {high_interest_flag}")
    st.caption("Note: Previous Loan Status is required because the trained model uses this feature.")

    if not submitted:
        st.info("Enter applicant details and click 'Score Applicant' to run prediction.")
        return

    input_payload = {
        "Age": age,
        "Income": income,
        "Home": home,
        "Emp_length": emp_length,
        "Intent": intent,
        "Amount": amount,
        "Rate": rate,
        "High_Interest_Flag": high_interest_flag,
        "Status": status,
        "Percent_income": percent_income,
    }

    input_df = pd.DataFrame([input_payload]).reindex(columns=expected_features)

    prediction = int(model.predict(input_df)[0])
    class_probs = model.predict_proba(input_df)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    default_idx = classes.index(1) if 1 in classes else len(classes) - 1
    default_probability = float(class_probs[default_idx])

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Predicted Outcome", DEFAULT_LABELS.get(prediction, str(prediction)))
    with result_col2:
        st.metric("Default Probability", f"{default_probability:.2%}")

    st.progress(min(max(default_probability, 0.0), 1.0))

    if default_probability >= 0.50:
        risk_band = "High"
        st.error("Risk Band: High")
    elif default_probability >= 0.25:
        risk_band = "Medium"
        st.warning("Risk Band: Medium")
    else:
        risk_band = "Low"
        st.success("Risk Band: Low")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_payload,
        "prediction": DEFAULT_LABELS.get(prediction, str(prediction)),
        "default_probability": f"{default_probability:.2%}",
        "default_probability_value": round(default_probability, 6),
        "risk_band": risk_band,
    }

    report_pdf = build_scoring_report_pdf(report)
    safe_ts = report["generated_at_utc"].replace(":", "").replace("-", "").split(".")[0]

    st.download_button(
        label="Download Scoring Report (PDF)",
        data=report_pdf,
        file_name=f"credit_scoring_report_{safe_ts}.pdf",
        mime="application/pdf",
    )

    with st.expander("View Model Input Payload"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)



def main() -> None:
    missing_files = [path.name for path in [DATA_PATH, MODEL_PATH] if not path.exists()]
    if missing_files:
        st.error(
            "Missing required artifacts: " + ", ".join(missing_files)
            + ". Place them in the same folder as credit_risk_app.py."
        )
        st.stop()

    try:
        df = load_dataset(DATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    missing_columns = find_missing_columns(df, REQUIRED_COLUMNS)
    if missing_columns:
        st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
        st.stop()

    try:
        model = load_pipeline(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Executive Summary", "Portfolio Analysis", "Applicant Scoring"],
    )

    if page == "Executive Summary":
        render_executive_summary(df)
    elif page == "Portfolio Analysis":
        render_portfolio_analysis(df)
    else:
        render_scoring(df, model)


if __name__ == "__main__":
    main()
