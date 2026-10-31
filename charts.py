from __future__ import annotations

import altair as alt
import pandas as pd

from inference import CLASS_NAMES, MODEL_COLORS, MODEL_SPECS


MODEL_SCALE = alt.Scale(domain=[s.label for s in MODEL_SPECS], range=MODEL_COLORS)


def style_chart(chart: alt.Chart, dark: bool, height: int = 255) -> alt.Chart:
    text = "#E9EEF2" if dark else "#334155"
    grid = "#32383E" if dark else "#E6EAEE"
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=text, titleColor=text, gridColor=grid, domain=False,
                            labelFontSize=12, titleFontSize=12, labelLimit=180, titlePadding=12)
            .configure_legend(labelColor=text, titleColor=text, orient="bottom", labelLimit=250)
            .configure_title(color=text, fontSize=16, anchor="start"))


def grouped_bars(frame: pd.DataFrame, value: str, dark: bool,
                 category: str = "Kelas", percent: bool = True,
                 model_order: list[str] | None = None) -> alt.Chart:
    order = model_order or [s.label for s in MODEL_SPECS]
    colors = dict(zip([s.label for s in MODEL_SPECS], MODEL_COLORS))
    model_scale = alt.Scale(domain=order, range=[colors[label] for label in order])
    scale = alt.Scale(domain=[0, 100]) if percent and "Confidence" in value else alt.Scale(zero=True)
    chart = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=22).encode(
        x=alt.X(f"{category}:N", sort=CLASS_NAMES if category == "Kelas" else None,
                title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Model:N", sort=order),
        y=alt.Y(f"{value}:Q", title=value, scale=scale),
        color=alt.Color("Model:N", scale=model_scale, title=None, sort=order),
        tooltip=["Model:N", f"{category}:N", alt.Tooltip(f"{value}:Q", format=".2f")],
    )
    labels = chart.mark_text(dy=-8, fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=".2f"),
        color=alt.value("#E9EEF2" if dark else "#334155"),
    ).transform_filter(alt.datum[value] > 0)
    return style_chart(chart + labels, dark)


def threshold_lines(frame: pd.DataFrame, value: str, dark: bool) -> alt.Chart:
    chart = alt.Chart(frame).mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5).encode(
        x=alt.X("confidence:Q", title="Confidence threshold", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{value}:Q", title=value, scale=alt.Scale(zero=True)),
        color=alt.Color("Model:N", scale=MODEL_SCALE, title=None),
        tooltip=["Model:N", alt.Tooltip("confidence:Q", format=".2f"),
                 alt.Tooltip(f"{value}:Q", format=".3f")],
    )
    return style_chart(chart, dark)


def threshold_heatmap(frame: pd.DataFrame, value: str, dark: bool,
                      domain_max: float) -> alt.Chart:
    base = alt.Chart(frame).encode(
        x=alt.X("confidence:O", title="Confidence threshold", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("iou_nms:O", title="IoU NMS", sort="descending"),
        tooltip=[alt.Tooltip("confidence:Q", format=".2f"), alt.Tooltip("iou_nms:Q", format=".2f"),
                 alt.Tooltip(f"{value}:Q", format=".3f")],
    )
    cells = base.mark_rect(stroke="#FFFFFF", strokeWidth=1).encode(
        color=alt.Color(f"{value}:Q", scale=alt.Scale(domain=[0, max(1, domain_max)],
                                                    range=["#E8F5F3", "#087F72"]), title=value)
    )
    labels = base.mark_text(fontSize=12).encode(
        text=alt.Text(f"{value}:Q", format=".0f" if "deteksi" in value else ".2f"),
        color=alt.condition(alt.datum[value] > domain_max * 0.5, alt.value("white"), alt.value("#173B36")),
    )
    return style_chart(cells + labels, dark, height=210)
