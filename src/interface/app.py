from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from src.cfg.manager import get_available_models
from src.model.engine.predict import PGenPredictor

# Add project root to path if running from src/interface
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Pharmagen AI", page_icon="💊", layout="wide", initial_sidebar_state="expanded"
)

# --- TRADUCCIONES ---
TR = {
    "en": {
        "sidebar_title": "Pharmagen Control",
        "select_lang": "Language / Idioma",
        "select_model": "Select Model",
        "load_btn": "Load / Switch Model",
        "model_info": "Model Info",
        "feat_list": "Feature List",
        "target_list": "Target List",
        "no_model_warn": "Please load a model first.",
        "main_title": "Pharmagen AI Platform",
        "sub_title": "Pharmacogenetic Prediction & Efficacy Analysis",
        "instruct_load": "👈 Select and Load a model from the sidebar to begin.",
        "tab1": "🧬 Single Prediction",
        "tab2": "📂 Batch Analysis (CSV)",
        "single_header": "Single Patient Analysis",
        "predict_btn": "Predict Outcome",
        "analyzing": "Analyzing genotype-drug interaction...",
        "success": "Analysis Complete",
        "results": "Prediction Results",
        "error_fail": "Prediction failed. Check input values.",
        "batch_header": "High-Throughput Analysis",
        "upload_label": "Upload Patient Cohort Data (CSV/TSV)",
        "loaded_rows": "Loaded {rows} rows.",
        "run_batch_btn": "Run Batch Prediction",
        "processing": "Processing {rows} samples...",
        "batch_success": "Batch Processing Complete!",
        "results_preview": "Results Preview",
        "download_btn": "Download Full Report",
        "footer": "© 2025 Pharmagen Project | Developed by Adrim Hamed Outmani",
        "load_success": "Loaded {model}!",
        "load_error": "Failed to load model: {error}",
        "detected": "detected",
    },
    "es": {
        "sidebar_title": "Control Pharmagen",
        "select_lang": "Idioma / Language",
        "select_model": "Seleccionar Modelo",
        "load_btn": "Cargar / Cambiar Modelo",
        "model_info": "Info del Modelo",
        "feat_list": "Lista de Variables (Features)",
        "target_list": "Lista de Objetivos (Targets)",
        "no_model_warn": "Por favor, carga un modelo primero.",
        "main_title": "Plataforma IA Pharmagen",
        "sub_title": "Predicción Farmacogenética y Análisis de Eficacia",
        "instruct_load": "👈 Selecciona y carga un modelo en la barra lateral para comenzar.",
        "tab1": "🧬 Predicción Individual",
        "tab2": "📂 Análisis por Lotes (CSV)",
        "single_header": "Análisis de Paciente Individual",
        "predict_btn": "Predecir Resultado",
        "analyzing": "Analizando interacción genotipo-fármaco...",
        "success": "Análisis Completado",
        "results": "Resultados de la Predicción",
        "error_fail": "Fallo en la predicción. Revisa los valores de entrada.",
        "batch_header": "Análisis de Alto Rendimiento",
        "upload_label": "Subir Datos de Cohorte (CSV/TSV)",
        "loaded_rows": "Cargadas {rows} filas.",
        "run_batch_btn": "Ejecutar Predicción por Lotes",
        "processing": "Procesando {rows} muestras...",
        "batch_success": "¡Procesamiento por Lotes Completado!",
        "results_preview": "Vista Previa de Resultados",
        "download_btn": "Descargar Informe Completo",
        "footer": "© 2025 Proyecto Pharmagen | Desarrollado por Adrim Hamed Outmani",
        "load_success": "¡Modelo {model} cargado!",
        "load_error": "Error al cargar modelo: {error}",
        "detected": "detectado(s)",
    },
}

# --- ESTILOS CSS ---
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
if "predictor" not in st.session_state:
    st.session_state["predictor"] = None
if "current_model" not in st.session_state:
    st.session_state["current_model"] = ""
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"  # Default language


def t(key):
    """Helper to get translation."""
    return TR[st.session_state["lang"]].get(key, key)


# --- SIDEBAR ---
with st.sidebar:
    # Language Selector (Top Priority)
    lang_option = st.radio(
        "🌐 Language / Idioma",
        ["🇬🇧 English", "🇪🇸 Español"],
        index=0 if st.session_state["lang"] == "en" else 1,
        horizontal=True,
    )
    st.session_state["lang"] = "en" if "English" in lang_option else "es"

    st.markdown("---")
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=80)
    st.title(t("sidebar_title"))

    available_models = get_available_models()

    if not available_models:
        st.error("No models found in models.toml!")
        st.stop()

    selected_model = st.selectbox(
        t("select_model"),
        available_models,
        index=0
        if not st.session_state["current_model"]
        else available_models.index(st.session_state["current_model"]),
    )

    # Load Model Button
    if st.button(t("load_btn"), type="primary"):
        with st.spinner(f"Loading {selected_model}..."):
            try:
                predictor = PGenPredictor(selected_model)
                st.session_state["predictor"] = predictor
                st.session_state["current_model"] = selected_model
                st.success(t("load_success").format(model=selected_model))
            except Exception as e:
                st.error(t("load_error").format(error=str(e)))

    st.markdown("---")
    st.markdown(f"### {t('model_info')}")
    if st.session_state["predictor"]:
        pred = st.session_state["predictor"]
        st.info(f"**Device:** {pred.device}")
        st.markdown(f"**Features:** {len(pred.feature_cols)}")
        with st.expander(t("feat_list")):
            st.write(pred.feature_cols)
        st.markdown(f"**Targets:** {len(pred.target_cols)}")
        with st.expander(t("target_list")):
            st.write(pred.target_cols)
    else:
        st.warning(t("no_model_warn"))

# --- MAIN AREA ---
st.markdown(f'<div class="main-header">{t("main_title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{t("sub_title")}</div>', unsafe_allow_html=True)

if not st.session_state["predictor"]:
    st.info(t("instruct_load"))
    st.stop()

predictor = st.session_state["predictor"]

tab1, tab2 = st.tabs([t("tab1"), t("tab2")])

# --- TAB 1: Single Prediction ---
with tab1:
    st.header(t("single_header"))

    with st.form("single_pred_form"):
        col1, col2 = st.columns(2)

        inputs = {}
        # Dynamic form generation
        for i, feature in enumerate(predictor.feature_cols):
            with col1 if i % 2 == 0 else col2:
                encoder = predictor.encoders.get(feature)
                if hasattr(encoder, "classes_") and len(encoder.classes_) < 100:
                    inputs[feature] = st.selectbox(f"{feature}", options=encoder.classes_)
                else:
                    inputs[feature] = st.text_input(f"{feature}", help=f"Value for {feature}")

        submitted = st.form_submit_button(t("predict_btn"))

        if submitted:
            with st.spinner(t("analyzing")):
                result = predictor.predict_single(inputs)

            if result:
                st.success(t("success"))
                st.markdown(f"### {t('results')}")

                # Display results
                r_cols = st.columns(len(result))
                for idx, (k, v) in enumerate(result.items()):
                    with r_cols[idx % 3]:
                        if isinstance(v, list):
                            st.metric(label=k, value=f"{len(v)} {t('detected')}", delta=None)
                            st.write(v)
                        else:
                            st.metric(label=k, value=str(v))
            else:
                st.error(t("error_fail"))

# --- TAB 2: Batch Analysis ---
with tab2:
    st.header(t("batch_header"))

    uploaded_file = st.file_uploader(t("upload_label"), type=["csv", "tsv", "txt"])

    if uploaded_file:
        try:
            sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
            df = pd.read_csv(uploaded_file, sep=sep)

            st.dataframe(df.head(), use_container_width=True)
            st.caption(t("loaded_rows").format(rows=len(df)))

            if st.button(t("run_batch_btn")):
                with st.spinner(t("processing").format(rows=len(df))):
                    results = predictor.predict_dataframe(df)

                if results:
                    res_df = pd.DataFrame(results)
                    st.success(t("batch_success"))

                    final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

                    st.markdown(f"### {t('results_preview')}")
                    st.dataframe(res_df.head(), use_container_width=True)

                    csv = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=t("download_btn"),
                        data=csv,
                        file_name=f"pharmagen_report_{st.session_state['current_model']}.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(t("footer"))
