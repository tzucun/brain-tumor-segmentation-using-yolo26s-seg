from __future__ import annotations

import html
import hashlib
import base64
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from charts import grouped_bars
from comparison import class_prediction_frame
from inference import (
    MODEL_SPECS, PredictionOutput, get_device, load_model, model_revision, predict_image,
)
from utils import image_to_png_bytes, load_image_from_bytes, safe_filename


st.set_page_config(
    page_title="Segmentasi Tumor Otak YOLO26s-Seg",
    layout="wide",
)


def apply_style(dark_mode: bool) -> None:
    if dark_mode:
        colors = {
            "app": "#0F172A",
            "sidebar": "#111827",
            "panel": "#172033",
            "panel_soft": "#1E293B",
            "text": "#F8FAFC",
            "muted": "#CBD5E1",
            "border": "#334155",
            "input": "#111827",
            "primary": "#2563EB",
            "primary_hover": "#1D4ED8",
        }
    else:
        colors = {
            "app": "#FFFFFF",
            "sidebar": "#F6F8FB",
            "panel": "#FFFFFF",
            "panel_soft": "#F1F5F9",
            "text": "#1F2937",
            "muted": "#64748B",
            "border": "#D8DEE9",
            "input": "#FFFFFF",
            "primary": "#375A7F",
            "primary_hover": "#2F4D6D",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colors["app"]};
            --sidebar-bg: {colors["sidebar"]};
            --panel-bg: {colors["panel"]};
            --panel-soft-bg: {colors["panel_soft"]};
            --text-main: {colors["text"]};
            --text-muted: {colors["muted"]};
            --border-color: {colors["border"]};
            --input-bg: {colors["input"]};
            --primary-color: {colors["primary"]};
            --primary-hover: {colors["primary_hover"]};
        }}

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {{
            background: var(--app-bg) !important;
            color: var(--text-main) !important;
        }}
        .main .block-container, [data-testid="stMainBlockContainer"] {{
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}
        .stApp h1 {{ font-size: 1.85rem; line-height: 1.3; }}
        .stApp h2 {{ font-size: 1.4rem; }}
        .stApp h3 {{ font-size: 1.15rem; }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp label,
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li {{
            color: var(--text-main) !important;
            letter-spacing: 0 !important;
        }}
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p,
        .stApp .stCaption, .stApp small {{
            color: var(--text-muted) !important;
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border-color);
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {{
            color: var(--text-main) !important;
        }}
        .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
            color: var(--text-muted) !important;
        }}

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {{
            background: var(--input-bg) !important;
            border-color: var(--border-color) !important;
        }}
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] p {{
            color: var(--text-main) !important;
        }}
        [data-testid="stFileUploaderDropzone"] small {{
            color: var(--text-muted) !important;
        }}
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] li {{
            background: var(--panel-bg) !important;
            border-color: var(--border-color) !important;
        }}
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileName"] * {{
            color: var(--text-main) !important;
            opacity: 1 !important;
        }}
        [data-testid="stFileUploader"] svg {{
            color: var(--text-muted) !important;
            fill: currentColor !important;
        }}

        .stButton button,
        .stDownloadButton button,
        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--primary-color) !important;
            border: 1px solid var(--primary-color) !important;
            border-radius: 6px !important;
            color: #FFFFFF !important;
        }}
        .stButton button *,
        .stDownloadButton button *,
        .stApp .stButton button p,
        .stApp .stDownloadButton button p,
        [data-testid="stFileUploaderDropzone"] button * {{
            color: #FFFFFF !important;
            opacity: 1 !important;
        }}
        .stButton button:hover,
        .stDownloadButton button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {{
            background: var(--primary-hover) !important;
            border-color: var(--primary-hover) !important;
        }}
        .stButton button:disabled,
        .stButton button:disabled *,
        .stApp .stButton button:disabled p {{
            background: var(--panel-soft-bg) !important;
            border-color: var(--border-color) !important;
            color: var(--text-muted) !important;
        }}

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {{
            background: var(--panel-bg) !important;
            border-color: var(--border-color) !important;
            color: var(--text-main) !important;
        }}
        [data-testid="stExpander"] summary * {{
            color: var(--text-main) !important;
        }}
        [data-testid="stMetric"] {{
            background: var(--panel-bg) !important;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: .75rem;
        }}
        [data-testid="stMetric"] * {{
            color: var(--text-main) !important;
        }}
        [data-testid="stAlert"] {{
            color: var(--text-main) !important;
        }}
        [data-testid="stAlert"] * {{
            color: inherit !important;
        }}
        [data-testid="stSlider"] * {{
            color: var(--text-main) !important;
        }}
        [data-baseweb="slider"] [role="slider"] {{
            background-color: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
        }}
        hr {{
            border-color: var(--border-color) !important;
        }}

        .theme-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--border-color);
            border-radius: 6px;
        }}
        .theme-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--panel-bg);
            color: var(--text-main);
        }}
        .theme-table th,
        .theme-table td {{
            padding: .7rem .8rem;
            border-bottom: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            color: var(--text-main) !important;
            text-align: left;
            white-space: normal;
        }}
        .theme-table th {{
            background: var(--panel-soft-bg);
            font-weight: 600;
        }}
        .theme-table tr:last-child td {{ border-bottom: 0; }}
        .theme-table th:last-child,
        .theme-table td:last-child {{ border-right: 0; }}
        [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
        .prediction-image {{
            display: block;
            width: 100%;
            aspect-ratio: 1;
            padding: 0;
            border: 0;
            border-radius: 8px;
            overflow: hidden;
            background: #000000;
            cursor: zoom-in;
        }}
        .prediction-image img {{
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            transition: transform 180ms ease-out;
        }}
        .prediction-image:focus-visible {{
            outline: 3px solid var(--primary-color);
            outline-offset: 3px;
        }}
        @media (hover: hover) and (pointer: fine) {{
            .prediction-image:hover img {{ transform: scale(1.05); }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .prediction-image img {{ transition: none; }}
        }}
        .image-lightbox {{
            width: min(1100px, calc(100vw - 32px));
            max-height: calc(100dvh - 32px);
            padding: 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--panel-bg);
            color: var(--text-main);
            overflow: auto;
        }}
        .image-lightbox::backdrop {{ background: rgb(0 0 0 / 80%); }}
        .image-lightbox-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        .image-lightbox-close {{
            flex: none;
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--panel-soft-bg);
            color: var(--text-main);
            cursor: pointer;
        }}
        .image-lightbox-close:focus-visible {{ outline: 3px solid var(--primary-color); }}
        .image-lightbox img {{
            display: block;
            width: 100%;
            height: auto;
            max-height: 78dvh;
            object-fit: contain;
            background: #000000;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title(
        "Segmentasi Tumor Otak pada Citra MRI Dataset BRISC2025 "
        "Menggunakan Arsitektur YOLO26s-Seg"
    )
    st.write(
        "Aplikasi ini menerapkan YOLO26s-Seg pada citra MRI dalam ruang lingkup dataset "
        "BRISC2025 untuk menampilkan kelas tumor, lokasi melalui bounding box, dan area "
        "tumor melalui segmentation mask pada kelas glioma, meningioma, dan pituitary."
    )


def format_metric(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%" if pd.notna(value) else "-"
    except (TypeError, ValueError):
        return "-"


def render_html_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    def format_cell(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ", ".join(html.escape(str(item)) for item in value)
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value))

    header_cells = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{format_cell(row[column])}</td>" for column in frame.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"""
        <div class="theme-table-wrap">
            <table class="theme-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


APP_MODELS = tuple(sorted(MODEL_SPECS, key=lambda spec: spec.id != "dengan_negative"))


def render_sidebar() -> float:
    st.sidebar.header("Pengaturan")
    conf = st.sidebar.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Batas skor keyakinan model untuk menampilkan deteksi. Berlaku pada kedua model.",
    )
    st.sidebar.caption(
        "Nilai lebih tinggi membuat deteksi lebih selektif, tetapi deteksi dengan skor "
        "lebih rendah dapat tidak ditampilkan. Confidence bukan ukuran ketepatan diagnosis."
    )
    st.sidebar.divider()
    st.sidebar.subheader("Informasi model")
    st.sidebar.write("YOLO26s-Seg")
    st.sidebar.caption("Kelas tumor: glioma, meningioma, dan pituitary.")
    return conf


def prediction_key(data: bytes, conf: float) -> tuple:
    revisions = []
    for spec in APP_MODELS:
        try:
            revisions.append(model_revision(spec.weights))
        except OSError:
            revisions.append(f"{spec.weights}:unavailable")
    return hashlib.sha256(data).hexdigest(), conf, tuple(revisions)


def predict_both(image: Any, conf: float) -> tuple[dict[str, PredictionOutput], dict[str, str]]:
    outputs, errors = {}, {}
    progress = st.progress(0, text="Menyiapkan prediksi...")
    for index, spec in enumerate(APP_MODELS):
        progress.progress(index / len(APP_MODELS), text=f"Memproses model: {spec.label.lower()}...")
        try:
            model = load_model(str(spec.weights), revision=model_revision(spec.weights))
            outputs[spec.id] = predict_image(model, image, conf, None, get_device())
        except Exception as exc:
            errors[spec.id] = str(exc)
        progress.progress((index + 1) / len(APP_MODELS))
    progress.empty()
    return outputs, errors


def render_prediction_image(image: Any, title: str, viewer_id: str) -> None:
    encoded = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    title = html.escape(title, quote=True)
    viewer_id = html.escape(viewer_id, quote=True)
    source = f"data:image/png;base64,{encoded}"
    st.html(
        f'<button type="button" class="prediction-image" popovertarget="{viewer_id}" '
        f'aria-label="Perbesar {title}" title="Perbesar {title}">'
        f'<img src="{source}" alt="{title}"></button>'
        f'<div id="{viewer_id}" class="image-lightbox" popover="auto" '
        f'role="dialog" aria-label="{title}">'
        f'<div class="image-lightbox-header"><span>{title}</span>'
        f'<button type="button" class="image-lightbox-close" popovertarget="{viewer_id}" '
        f'popovertargetaction="hide" aria-label="Tutup gambar">Tutup</button></div>'
        f'<img src="{source}" alt="{title} dalam tampilan besar"></div>'
    )


def render_prediction_charts(outputs: dict[str, PredictionOutput], dark: bool) -> None:
    if not outputs:
        return
    frame = class_prediction_frame(outputs)
    order = [spec.label for spec in APP_MODELS]
    st.subheader("Visualisasi hasil prediksi")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confidence tertinggi per kelas**")
        scores = frame.dropna(subset=["Confidence (%)"])
        if scores.empty:
            st.info("Tidak ada deteksi yang melewati confidence threshold.")
        else:
            st.altair_chart(
                grouped_bars(scores, "Confidence (%)", dark, model_order=order),
                theme=None, use_container_width=True,
            )
    with right:
        st.markdown("**Luas mask per kelas**")
        areas = frame.dropna(subset=["Luas mask (%)"])
        if areas.empty:
            st.info("Mask segmentasi tidak tersedia.")
        elif not areas["Luas mask (%)"].gt(0).any():
            st.info("Tidak ada area tumor yang tersegmentasi pada citra ini.")
        else:
            st.altair_chart(
                grouped_bars(areas, "Luas mask (%)", dark, model_order=order),
                theme=None, use_container_width=True,
            )
    st.caption(
        "Confidence adalah skor deteksi, bukan ketepatan diagnosis. Kelas tanpa deteksi "
        "tidak memiliki skor confidence. Luas mask dihitung terhadap luas citra 2D."
    )


def render_prediction(result: dict[str, Any], dark: bool = False) -> None:
    outputs, errors = result["outputs"], result["errors"]
    st.subheader("Hasil prediksi")
    columns = st.columns(3)
    with columns[0]:
        st.markdown("**Citra masukan**")
        render_prediction_image(result["image"], "Citra masukan", "viewer-input")
    for column, spec in zip(columns[1:], APP_MODELS):
        with column:
            st.markdown(f"**{spec.label}**")
            output = outputs.get(spec.id)
            if output is None:
                st.error(f"Prediksi belum tersedia: {errors.get(spec.id, 'Model gagal diproses.')}")
            else:
                render_prediction_image(output.annotated_image, spec.label, f"viewer-{spec.id}")

    with st.expander("Detail Hasil Prediksi dan Visualisasi", expanded=False):
        for column, spec in zip(st.columns(2), APP_MODELS):
            with column:
                output = outputs.get(spec.id)
                if output is None:
                    continue
                st.markdown(f"**{spec.label}**")
                if output.detections:
                    st.write(f"Hasil utama: **{output.primary_class}**")
                    frame = pd.DataFrame(output.detections).rename(columns={
                        "kelas": "Kelas", "confidence": "Confidence",
                        "bbox": "Bounding box (x1, y1, x2, y2)", "luas_mask_px": "Luas mask (piksel)",
                    })
                    render_html_table(frame)
                else:
                    st.info("Tidak ada prediksi tumor yang melewati confidence threshold.")
                if output.masks_available:
                    st.write(f"Perkiraan luas total mask terhadap citra: **{output.mask_area_percent:.2f}%**")
                else:
                    st.warning("Mask segmentasi tidak tersedia; luas mask tidak dapat dihitung.")
                st.write(f"Waktu inferensi: **{output.inference_time:.3f} detik**")
                filename = f"{Path(safe_filename(result['file_name'])).stem}_{spec.id}.png"
                st.download_button(
                    "Unduh gambar hasil prediksi", image_to_png_bytes(output.annotated_image),
                    file_name=filename, mime="image/png", key=f"download_{spec.id}", on_click="ignore",
                )
        render_prediction_charts(outputs, dark)


def render_research_information() -> None:
    with st.expander("Kinerja Model pada Dataset Uji", expanded=False):
        st.caption(
            "Metrik evaluasi model dari hasil penelitian. Pengaturan confidence untuk "
            "citra unggahan tidak mengubah nilai metrik berikut."
        )
        fields = [
            ("box_mAP50", "box mAP50"), ("box_mAP50_95", "box mAP50-95"),
            ("mask_mAP50", "mask mAP50"), ("mask_mAP50_95", "mask mAP50-95"),
        ]
        for spec in APP_MODELS:
            st.markdown(f"**Model {spec.label.lower()}**")
            st.caption(spec.description)
            try:
                summary = pd.read_csv(spec.summary)
                if summary.empty:
                    st.info("Ringkasan metrik belum tersedia.")
                else:
                    for column, (field, label) in zip(st.columns(4), fields):
                        column.metric(label, format_metric(summary.iloc[0].get(field)))
            except (OSError, ValueError) as exc:
                st.info(f"Ringkasan metrik belum dapat dibaca: {exc}")
            try:
                per_class = pd.read_csv(spec.per_class)
                selected = [name for name in ("class_name", "box_mAP50_95", "mask_mAP50_95")
                            if name in per_class.columns]
                if selected:
                    frame = per_class[selected].copy()
                    for field in ("box_mAP50_95", "mask_mAP50_95"):
                        if field in frame:
                            frame[field] = frame[field].map(format_metric)
                    render_html_table(frame.rename(columns={
                        "class_name": "Kelas", "box_mAP50_95": "box mAP50-95",
                        "mask_mAP50_95": "mask mAP50-95",
                    }))
            except (OSError, ValueError) as exc:
                st.info(f"Metrik per kelas belum dapat dibaca: {exc}")
        st.caption(
            "Negative samples adalah citra no_tumor dengan label kosong saat pelatihan, "
            "bukan kelas tumor keempat. Luas mask merupakan proporsi area pada citra 2D, "
            "bukan volume tumor."
        )


def render_footer() -> None:
    st.divider()
    st.caption(
        "Aplikasi ini ditujukan sebagai alat bantu bagi tenaga medis untuk memperoleh "
        "gambaran awal mengenai kelas, lokasi, dan area tumor pada citra MRI. "
        "Hasil prediksi dapat mendukung proses diagnosis awal, tetapi tidak menggantikan "
        "pemeriksaan klinis dan penetapan diagnosis oleh tenaga medis."
    )


def run_app() -> None:
    dark = st.sidebar.toggle("Mode gelap", value=False)
    apply_style(dark)
    conf = render_sidebar()
    render_header()
    uploaded = st.file_uploader("Unggah citra MRI", type=["jpg", "jpeg", "png"])
    data = uploaded.getvalue() if uploaded is not None else None
    image = None
    if data is not None:
        try:
            image = load_image_from_bytes(data)
        except Exception as exc:
            st.error(f"Berkas citra tidak dapat dibaca: {exc}")
    run = st.button("Jalankan prediksi", type="primary", disabled=image is None)
    key = prediction_key(data, conf) if image is not None else None
    if run:
        outputs, errors = predict_both(image, conf)
        st.session_state["prediction"] = {
            "key": key, "image": image, "file_name": uploaded.name,
            "outputs": outputs, "errors": errors,
        }
    result = st.session_state.get("prediction")
    if result is not None and key is not None:
        if result["key"] == key:
            render_prediction(result, dark)
        else:
            st.info("Citra atau confidence berubah. Tekan Jalankan prediksi untuk memperbarui hasil.")
    render_research_information()
    render_footer()


if __name__ == "__main__":
    run_app()
