import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from agents import AgenteMaestro

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Framework",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262b3d);
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 4px solid #4f8bf9;
        margin-bottom: 10px;
    }
    .metric-card.green  { border-left-color: #00c49a; }
    .metric-card.red    { border-left-color: #ff4b6e; }
    .metric-card.yellow { border-left-color: #f9a825; }
    .metric-label { font-size: 12px; color: #9ca3af; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #f0f2f6; margin-top: 4px; }
    .metric-sub   { font-size: 12px; color: #6b7280; margin-top: 2px; }
    .section-title {
        font-size: 20px; font-weight: 700; color: #e2e8f0;
        border-bottom: 2px solid #4f8bf9;
        padding-bottom: 6px; margin-bottom: 16px;
    }
    [data-testid="stSidebar"] { background-color: #13151f; }
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def metric_card(label, value, sub="", color="blue"):
    return f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


def score_color(score):
    if score >= 0.75:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def fmt_pct(v):
    return f"{v*100:.1f}%"


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ── Sidebar: configuración global ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuracion global")
    st.markdown("---")
    test_size = st.slider("Tamaño de test (%)", 10, 40, 25)
    random_seed = st.number_input("Semilla aleatoria", value=42, step=1)


# ── Session state para el dataset ────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "maestro" not in st.session_state:
    st.session_state.maestro = None
if "sugerencia" not in st.session_state:
    st.session_state.sugerencia = None


def _cargar_dataset(uploaded_file, sep, use_dummies):
    """Carga el CSV, aplica opciones y guarda en session_state."""
    if uploaded_file is None:
        st.session_state.df = None
        st.session_state.maestro = None
        st.session_state.sugerencia = None
        return
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        if use_dummies:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                df = df.astype({c: int for c in df.select_dtypes(include='bool').columns})
        st.session_state.df = df
        st.session_state.maestro = AgenteMaestro(df)
        st.session_state.sugerencia = st.session_state.maestro.sugerir_tipo()
    except Exception as e:
        import traceback; traceback.print_exc()
        st.warning(f"ADVERTENCIA: falló sugerir_tipo — {e}")
        if 'df' in dir():
            st.session_state.df = df


# ── Tabs principales ─────────────────────────────────────────────────────────
tab_inicio, tab_carga, tab_eda, tab_sup, tab_ns, tab_dash = st.tabs([
    "Inicio", "Carga de Datos", "EDA", "Supervisado", "No supervisado", "Dashboard Ejecutivo",
])

with tab_sup:
    st.caption("Modelos que requieren una variable objetivo (target).")
    tab_clf, tab_reg = st.tabs(["Clasificación", "Regresión"])

with tab_ns:
    st.caption("Modelos que no requieren etiquetas — descubren estructura en los datos.")
    (tab_cluster,) = st.tabs(["Clustering"])


# ── Variables de acceso rápido ────────────────────────────────────────────────
df = st.session_state.df
maestro = st.session_state.maestro
sugerencia = st.session_state.sugerencia
dataset_cargado = df is not None

if dataset_cargado and sugerencia:
    _iconos = {"clasificacion": "", "regresion": "", "no_supervisado": ""}
    _etiquetas = {"clasificacion": "Clasificación", "regresion": "Regresión",
                  "no_supervisado": "No supervisado (Clustering / EDA)"}
    _icono = _iconos.get(sugerencia["tipo"], "")
    _etiqueta = _etiquetas.get(sugerencia["tipo"], sugerencia["tipo"])

    st.markdown(
        f"""
        <div style="background:#1a2332;border-left:4px solid #4f8bf9;
                    padding:14px 18px;border-radius:8px;margin-bottom:18px;">
            <div style="color:#9ca3af;font-size:12px;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.05em;">
                Sugerencia automática
            </div>
            <div style="color:#f0f2f6;font-size:18px;font-weight:700;margin-top:4px;">
                {_icono} {_etiqueta}
                {f"· target: <code>{sugerencia['target_sugerido']}</code>"
                 if sugerencia["target_sugerido"] else ""}
            </div>
            <div style="color:#9ca3af;font-size:13px;margin-top:6px;">
                {sugerencia["razon"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Inicializar auto-sugerencia cuando se carga un dataset ───────────────────
if dataset_cargado and sugerencia is None and maestro is not None:
    sugerencia = maestro.sugerir_tipo()
    st.session_state.sugerencia = sugerencia


# ════════════════════════════════════════════════════════════════════════════
# TAB 0 · INICIO
# ════════════════════════════════════════════════════════════════════════════
with tab_inicio:
    st.markdown("""
    <div style="background: linear-gradient(135deg,#1a1f35,#1e3a5f);
                border-radius:16px; padding:36px 40px; margin-bottom:24px;
                text-align:center;">
        <h1 style="margin:0;color:#f0f2f6;font-size:40px;font-weight:800;">
            ML Framework
        </h1>
        <p style="margin:12px 0 0;color:#9ca3af;font-size:18px;max-width:700px;margin-left:auto;margin-right:auto;line-height:1.6;">
            Sistema Multiagente Inteligente para la Automatización de Procesos de Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Información del Proyecto</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        for label, value in [
            ("Curso", "EIF420O - Inteligencia Artificial"),
            ("Ciclo", "I Ciclo 2026"),
            ("Profesor", "Dr. Juan De Dios Murillo Morera"),
            ("Escuela", "Escuela de Informática"),
        ]:
            st.markdown(f"""
            <div style="background:#1a2332;border-radius:10px;padding:14px 18px;margin-bottom:10px;">
                <div style="color:#9ca3af;font-size:11px;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.05em;">{label}</div>
                <div style="color:#f0f2f6;font-size:16px;font-weight:600;margin-top:4px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Integrantes</div>', unsafe_allow_html=True)
        for nombre in ["Santiago Azofeifa Benavides", "Rubén Ramos Jiménez", "Gabriel Varela Araya"]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:8px 0;">
                <span style="color:#e2e8f0;font-size:15px;font-weight:500;">{nombre}</span>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════
with tab_carga:
    st.markdown('<div class="section-title">Carga de Datos</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"],
                                         key="carga_upload")
    with cc2:
        sep = st.radio("Separador", [",", ";"], horizontal=True, key="carga_sep")
        use_dummies = st.checkbox("Convertir categóricas a dummies", value=False,
                                  key="carga_dummies",
                                  help="Aplica get_dummies a las columnas categóricas")

    if uploaded_file is not None:
        _prev = st.session_state.get("_carga_params")
        _cur = (uploaded_file.name, sep, use_dummies)
        if st.session_state.df is None or _prev != _cur:
            for k in ["df", "maestro", "sugerencia", "cluster_result", "_carga_params"]:
                st.session_state[k] = None
            _cargar_dataset(uploaded_file, sep, use_dummies)
            st.session_state._carga_params = _cur
        df_tmp = st.session_state.df
        if df_tmp is not None:
            st.success(f"Dataset cargado: **{uploaded_file.name}** — "
                       f"{df_tmp.shape[0]} filas × {df_tmp.shape[1]} columnas")
            st.markdown("---")
            st.markdown("**Vista previa (primeras 10 filas)**")
            st.dataframe(df_tmp.head(10), width='stretch')

            st.markdown("---")
            st.markdown("**Información del dataset**")
            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.metric("Filas", df_tmp.shape[0])
            ci2.metric("Columnas", df_tmp.shape[1])
            ci3.metric("Numéricas", len(df_tmp.select_dtypes(include="number").columns))
            ci4.metric("No numéricas", len(df_tmp.select_dtypes(exclude="number").columns))

            st.markdown("**Tipos de datos**")
            tipos = pd.DataFrame({
                "Columna": df_tmp.dtypes.index,
                "Tipo": df_tmp.dtypes.values.astype(str),
                "Valores nulos": df_tmp.isnull().sum().values,
                "% Nulos": (df_tmp.isnull().sum() / len(df_tmp) * 100).round(1).values,
                "Valores únicos": df_tmp.nunique().values,
            }).reset_index(drop=True)
            st.dataframe(tipos, width='stretch', hide_index=True)
    else:
        st.info("Sube un archivo CSV para comenzar el análisis.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    df = st.session_state.df
    dataset_cargado = df is not None
    if not dataset_cargado:
        st.info(" Sube un archivo CSV desde la barra lateral para ver el análisis exploratorio.")
    else:
        maestro = st.session_state.maestro
        sugerencia = st.session_state.sugerencia
        st.markdown('<div class="section-title">Resumen del Dataset</div>', unsafe_allow_html=True)

        resumen = maestro.ejecutar_eda()
        num_cols = resumen["num_cols"]
        cat_cols = resumen["cat_cols"]

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Filas", f"{resumen['shape'][0]:,}", "registros totales"), unsafe_allow_html=True)
        c2.markdown(metric_card("Columnas", resumen["shape"][1],
                                f"{len(num_cols)} num · {len(cat_cols)} cat"), unsafe_allow_html=True)
        c3.markdown(metric_card("Valores nulos", resumen["nulos_totales"], "en todo el dataset",
                                "red" if resumen["nulos_totales"] else "green"), unsafe_allow_html=True)
        c4.markdown(metric_card("Duplicados", resumen["duplicados"], "filas duplicadas",
                                "red" if resumen["duplicados"] else "green"), unsafe_allow_html=True)

        st.markdown("---")

        col_prev, col_types = st.columns([2, 1])
        with col_prev:
            st.markdown("**Vista previa**")
            st.dataframe(df.head(8), width='stretch')
        with col_types:
            st.markdown("**Tipos de datos**")
            dtype_df = pd.DataFrame({"Columna": df.columns,
                                     "Tipo": df.dtypes.astype(str).values,
                                     "Nulos": df.isnull().sum().values})
            st.dataframe(dtype_df, width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Estadísticas Descriptivas</div>', unsafe_allow_html=True)
        st.dataframe(resumen["descripcion"].T.style.format(precision=3), width='stretch')

        with st.expander("Estadísticas avanzadas (moda, varianza, rango)"):
            est_avanzado = maestro._eda.analisis_estadistico()
            if est_avanzado:
                df_est = pd.DataFrame(est_avanzado).T
                st.dataframe(
                    df_est.style.format(precision=3)
                        .background_gradient(subset=["media", "mediana", "varianza", "desv_std"], cmap="Blues"),
                    width='stretch'
                )
            else:
                st.info("No hay columnas numéricas para mostrar estadísticas avanzadas.")

        st.markdown("---")
        st.markdown('<div class="section-title">Visualizaciones</div>', unsafe_allow_html=True)

        viz_type = st.selectbox("Tipo de gráfico", [
            "Boxplots", "Histogramas", "Distribución (KDE)", "Densidad",
            "Correlaciones (Pearson)", "Correlaciones (Spearman)", "Dispersión por pares"
        ])

        if num_cols:
            try:
                if viz_type == "Boxplots":
                    show_fig(maestro._eda.grafico_boxplot())
                elif viz_type == "Histogramas":
                    show_fig(maestro._eda.histogramas())
                elif viz_type == "Distribución (KDE)":
                    show_fig(maestro._eda.distribucion_variables())
                elif viz_type == "Densidad":
                    show_fig(maestro._eda.grafico_densidad())
                elif viz_type == "Correlaciones (Pearson)":
                    show_fig(maestro._eda.grafico_correlacion())
                elif viz_type == "Correlaciones (Spearman)":
                    fig = maestro._eda.grafico_correlacion_spearman()
                    if fig is not None:
                        show_fig(fig)
                elif viz_type == "Dispersión por pares":
                    if len(num_cols) <= 6:
                        fig = maestro._eda.graficos_dispersion()
                        if fig is not None:
                            show_fig(fig)
                    else:
                        sel = st.multiselect("Selecciona hasta 5 columnas", num_cols, num_cols[:5])
                        if len(sel) >= 2:
                            fig = maestro._eda.graficos_dispersion(columnas=sel)
                            if fig is not None:
                                show_fig(fig)
            except Exception as e:
                st.error(f"Error generando gráfico: {e}")

        if num_cols:
            st.markdown("---")
            st.markdown('<div class="section-title">Detección de Outliers</div>', unsafe_allow_html=True)

            out_method = st.radio("Método", ["IQR (Rango Intercuartil)", "Z-Score"], horizontal=True)
            out_umbral = st.slider("Umbral Z-Score", 2.0, 5.0, 3.0, 0.5) if out_method == "Z-Score" else None

            if st.button("Detectar outliers", type="primary", key="btn_outliers"):
                with st.spinner("Analizando outliers..."):
                    if out_method == "IQR (Rango Intercuartil)":
                        res_out = maestro._eda.detectar_outliers_iqr()
                    else:
                        res_out = maestro._eda.detectar_outliers_zscore(umbral=out_umbral)

                if res_out:
                    total_out = sum(r["n"] for r in res_out.values())
                    cols_con = [c for c, r in res_out.items() if r["n"] > 0]

                    o1, o2 = st.columns(2)
                    o1.markdown(metric_card("Columnas con outliers", len(cols_con),
                                            f"de {len(res_out)} numéricas", "red" if cols_con else "green"),
                                unsafe_allow_html=True)
                    o2.markdown(metric_card("Total outliers", total_out, "en todo el dataset",
                                            "red" if total_out else "green"), unsafe_allow_html=True)

                    tabla_out = pd.DataFrame([{
                        "Columna": c,
                        "Outliers": r["n"],
                        "% del total": f"{r['pct']:.1f}%",
                        "Límite inferior": r.get("limite_inf", "-"),
                        "Límite superior": r.get("limite_sup", "-"),
                    } for c, r in res_out.items()])
                    st.dataframe(tabla_out, width='stretch', hide_index=True)

                    if out_method == "IQR (Rango Intercuartil)":
                        fig_out = maestro._eda.grafico_outliers_iqr(res_out)
                        if fig_out is not None:
                            st.markdown("**Boxplots con límites IQR**")
                            show_fig(fig_out)

                    for col in cols_con[:3]:
                        r = res_out[col]
                        vals = r["outliers"].head(10)
                        with st.expander(f"Outliers en '{col}' ({r['n']} registros)"):
                            st.dataframe(pd.DataFrame({col: vals}), width='stretch')
                else:
                    st.info("No se encontraron outliers con el método seleccionado.")

        if cat_cols:
            st.markdown("---")
            st.markdown('<div class="section-title">Columnas Categóricas</div>', unsafe_allow_html=True)
            col_sel = st.selectbox("Ver distribución de", cat_cols)
            vc = df[col_sel].value_counts().reset_index()
            vc.columns = [col_sel, "Frecuencia"]
            vc["Porcentaje"] = (vc["Frecuencia"] / vc["Frecuencia"].sum() * 100).round(1).astype(str) + "%"

            cc1, cc2 = st.columns([1, 2])
            with cc1:
                st.dataframe(vc, width='stretch', hide_index=True)
            with cc2:
                show_fig(maestro._eda.histograma_clase(col_sel))

        st.markdown("---")
        st.markdown('<div class="section-title">Calidad de Datos</div>', unsafe_allow_html=True)

        inconsistencias = maestro._eda.detectar_inconsistencias()
        if inconsistencias:
            st.markdown("**Registros inconsistentes detectados**")
            df_inc = pd.DataFrame([{
                "Tipo": i["tipo"].replace("_", " ").title(),
                "Columna": i["columna"],
                "Descripción": i["descripcion"],
            } for i in inconsistencias])
            st.dataframe(df_inc, width='stretch', hide_index=True)

            for inc in inconsistencias:
                if inc["indices"] and len(inc["indices"]) <= 10:
                    with st.expander(f"Detalle: {inc['columna']}"):
                        st.write(f"Filas afectadas: {inc['indices']}")
        else:
            st.success("No se detectaron registros inconsistentes en el dataset.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · CLUSTERING (usa Clustering del framework)
# ════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    df = st.session_state.df
    dataset_cargado = df is not None
    if not dataset_cargado:
        st.info(" Sube un archivo CSV desde la barra lateral para ejecutar clustering.")
    else:
        st.markdown('<div class="section-title">Configuración del Clustering</div>', unsafe_allow_html=True)

        df_num_cl = df.select_dtypes(include="number").dropna()
        if df_num_cl.empty or df_num_cl.shape[1] < 2:
            st.warning("Se necesitan al menos 2 columnas numéricas para clustering.")

        cl1, cl2 = st.columns([1, 2])
        with cl1:
            algos_disponibles = ["KMeans", "KMedoids", "HAC", "DBSCAN", "T-SNE", "UMAP", "Comparar todos"]
            algo_cl = st.selectbox("Algoritmo", algos_disponibles)

            es_individual = algo_cl != "Comparar todos"
            if es_individual:
                if algo_cl == "DBSCAN":
                    n_clusters_cl = None
                    st.caption("DBSCAN detecta el número de clusters automáticamente.")
                else:
                    n_clusters_cl = st.slider("Número de clusters", 2, 10, 3)
                normalizar = st.checkbox("Normalizar (StandardScaler)", value=True)
                if algo_cl == "HAC":
                    linkage_method = st.selectbox("Método de linkage", ["ward", "average", "single", "complete"])
                else:
                    linkage_method = "ward"
                if algo_cl == "KMedoids":
                    metric_km = st.selectbox("Métrica de distancia", ["euclidean", "cityblock"])
                else:
                    metric_km = "euclidean"
                if algo_cl == "DBSCAN":
                    eps_db = st.slider("eps (radio de vecindad)", 0.1, 5.0, 0.5, 0.1)
                    min_samples_db = st.slider("min_samples", 2, 20, 5)
                else:
                    eps_db, min_samples_db = 0.5, 5
            else:
                n_clusters_cl = st.slider("Número de clusters", 2, 10, 3)
                normalizar = st.checkbox("Normalizar (StandardScaler)", value=True)
                linkage_method = "ward"
                metric_km = "euclidean"
                eps_db, min_samples_db = 0.5, 5
        with cl2:
            st.markdown("**Columnas disponibles**")
            cols_sel = st.multiselect("Usar columnas", df_num_cl.columns.tolist(),
                                      df_num_cl.columns.tolist())

        if not cols_sel or len(cols_sel) < 2:
            st.warning("Selecciona al menos 2 columnas.")

        # ── Estado para persistir resultados fuera del botón ────────────────
        if "cluster_result" not in st.session_state:
            st.session_state.cluster_result = None

        if st.button("Ejecutar clustering" if es_individual else "Comparar todos los algoritmos",
                     type="primary", key="btn_cluster"):
            try:
                df_cluster = df[cols_sel].dropna()
                maestro_cluster = AgenteMaestro(df_cluster)

                if es_individual:
                    with st.spinner(f"Entrenando {algo_cl}..."):
                        kwargs = {}
                        if algo_cl == "HAC":
                            kwargs = {"linkage": linkage_method}
                        elif algo_cl == "KMedoids":
                            kwargs = {"metric": metric_km}
                        elif algo_cl == "DBSCAN":
                            kwargs = {"eps": eps_db, "min_samples": min_samples_db}
                            n_clusters_cl = None

                        res = maestro_cluster.ejecutar_clustering(
                            algoritmo=algo_cl,
                            n_clusters=n_clusters_cl if n_clusters_cl else 3,
                            normalizar=normalizar,
                            random_state=int(random_seed),
                            **kwargs,
                        )
                        st.session_state.cluster_result = {
                            "modo": "individual",
                            "algo": algo_cl,
                            "linkage": linkage_method,
                            "res": res,
                        }
                else:
                    with st.spinner("Comparando algoritmos de clustering..."):
                        resultados, tabla = maestro_cluster.comparar_clustering(
                            n_clusters=n_clusters_cl,
                            columnas=cols_sel,
                            normalizar=normalizar,
                            random_state=int(random_seed),
                        )
                        mejor = maestro_cluster.decidir_clustering(resultados)
                        st.session_state.cluster_result = {
                            "modo": "comparar",
                            "resultados": resultados,
                            "tabla": tabla,
                            "mejor": mejor,
                        }
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.cluster_result = None
                st.stop()

        # ── Mostrar resultados desde session_state ──────────────────────────
        from core.clustering import Clustering as Cls
        cr = st.session_state.cluster_result
        if cr is not None and cr["modo"] == "individual" and cr["algo"] == algo_cl:
            res = cr["res"]
            mtr = res["metricas"]
            st.markdown("---")
            st.markdown('<div class="section-title">Métricas del Modelo</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(metric_card("Algoritmo", res["algoritmo"], ""), unsafe_allow_html=True)
            m2.markdown(metric_card("Silhouette", f"{mtr['silhouette']:.3f}",
                                    "Rango: -1 a 1", score_color(mtr["silhouette"])),
                        unsafe_allow_html=True)
            m3.markdown(metric_card("Calinski-Harabasz", f"{mtr['calinski_harabasz']:.1f}",
                                    "Cuanto mayor, mejor"), unsafe_allow_html=True)
            m4.markdown(metric_card("Davies-Bouldin", f"{mtr['davies_bouldin']:.3f}",
                                    "Cuanto menor, mejor",
                                    "green" if mtr["davies_bouldin"] < 1 else "yellow"),
                        unsafe_allow_html=True)

            st.markdown("---")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Proyección 2D**")
                fig = Cls.plot_proyeccion_2d_fig(
                    res["X_proj"], res["labels"], res["n_clusters"],
                    titulo=f"{res['algoritmo']} — {res['n_clusters']} clusters",
                )
                show_fig(fig)
            with g2:
                st.markdown("**Distribución por cluster**")
                fig = Cls.plot_distribucion_clusters_fig(res["labels"], res["n_clusters"])
                show_fig(fig)

            st.markdown("---")
            st.markdown("**Perfil de cada cluster (media de variables)**")
            df_perfil = res["df_original"].copy()
            df_perfil["Cluster"] = res["labels"]
            perfil = df_perfil.groupby("Cluster").mean().round(3)
            st.dataframe(
                perfil.style.background_gradient(cmap="Blues", axis=0).format(precision=3),
                width='stretch'
            )

            if algo_cl == "HAC" and len(res["X"]) <= 200:
                st.markdown("---")
                st.markdown(f"**Dendrograma - método: {linkage_method}**")
                fig = Cls.plot_dendrograma_fig(res["X"], metodo=linkage_method)
                show_fig(fig)

            if len(res["X"]) <= 2000:
                st.markdown("---")
                st.markdown("**Determinación del número óptimo de clusters**")
                metodo_k = st.radio("Método", ["Silhouette Score", "Método del Codo (Inercia)"],
                                    horizontal=True, key="metodo_k")
                X_for_curve = res["X_proj"] if algo_cl in ("T-SNE", "UMAP") else res["X"]
                if metodo_k == "Silhouette Score":
                    fig, best_k, best_score = Cls.plot_silhouette_curve_fig(
                        X_for_curve, k_min=2, k_max=10, random_state=int(random_seed)
                    )
                    show_fig(fig)
                    st.info(f"Mejor k según Silhouette: **k = {best_k}** (score = {best_score:.3f})")
                else:
                    fig, ks, inercias = Cls.plot_codo_fig(
                        X_for_curve, k_min=2, k_max=10, random_state=int(random_seed)
                    )
                    show_fig(fig)
                    diffs = np.diff(inercias, 2)
                    codo_k = ks[np.argmin(diffs) + 1] if len(diffs) > 0 else ks[-1]
                    st.info(f"Codo estimado en **k = {codo_k}** (punto de máxima curvatura)")

        elif cr is not None and cr["modo"] == "comparar":
            tabla = cr["tabla"]
            mejor = cr["mejor"]
            st.markdown("---")
            st.markdown('<div class="section-title">Comparación de Clustering</div>', unsafe_allow_html=True)

            st.dataframe(
                tabla.style.format(precision=3)
                    .background_gradient(subset=["Silhouette"], cmap="Greens")
                    .background_gradient(subset=["Davies-Bouldin"], cmap="Reds_r"),
                width='stretch', hide_index=True
            )

            st.info(f"Mejor algoritmo segun el agente: **{mejor['algoritmo']}** "
                f"(Silhouette={mejor['metricas'].get('silhouette', float('nan')):.3f}, "
                f"CH={mejor['metricas'].get('calinski_harabasz', float('nan')):.1f}, "
                f"DB={mejor['metricas'].get('davies_bouldin', float('nan')):.3f})")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(tabla["Algoritmo"], tabla["Silhouette"],
                    color=sns.color_palette("Set2", len(tabla)),
                    edgecolor="white", linewidth=0.5)
            for i, val in enumerate(tabla["Silhouette"]):
                ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)
            ax.set_xlabel("Silhouette")
            ax.set_title("Silhouette por algoritmo")
            ax.invert_yaxis()
            plt.tight_layout()
            show_fig(fig)




# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · CLASIFICACIÓN
# ════════════════════════════════════════════════════════════════════════════
with tab_clf:
    df = st.session_state.df
    dataset_cargado = df is not None
    if not dataset_cargado:
        st.info(" Sube un archivo CSV desde la barra lateral para ejecutar clasificación.")
    else:
        sugerencia = st.session_state.sugerencia
        st.markdown('<div class="section-title">Configuración</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            columnas_categoricas = []
            for col in df.columns:
                n_unique = df[col].nunique()
                if n_unique <= 50:
                    columnas_categoricas.append(col)
            
            _clf_opts = [None] + columnas_categoricas
            _clf_default = (_clf_opts.index(sugerencia["target_sugerido"])
                            if sugerencia is not None and sugerencia["tipo"] == "clasificacion"
                            and sugerencia["target_sugerido"] in columnas_categoricas else 0)
            target_clf = st.selectbox("Columna objetivo (target)", _clf_opts,
                                      index=_clf_default, key="tgt_clf",
                                      help="Solo columnas con <= 50 valores únicos (categóricas).")
        with cc2:
            algo_clf = st.selectbox("Algoritmo", [
                "KNN", "Decision Tree", "Random Forest", "SVM",
                "Logistic Regression", "Naive Bayes", "XGBoost",
                "Gradient Boosting", "AdaBoost", "Comparar todos"
            ], key="alg_clf")

        if target_clf is None:
            st.info("Selecciona la columna target para continuar.")

        with st.expander(" Parámetros del modelo"):
            p1, p2, p3 = st.columns(3)
            n_neighbors_v = p1.slider("KNN - vecinos", 1, 20, 5)
            max_depth_v = p2.slider("Max depth", 1, 15, 4)
            n_est_v = p3.slider("N estimadores", 10, 300, 100, 10)

            p4, p5, p6 = st.columns(3)
            c_reg_v = p4.slider("C (SVM / LR)", 0.01, 10.0, 1.0, 0.1)
            kernel_svm_v = p5.selectbox("Kernel (SVM)", ["rbf", "linear", "poly", "sigmoid"],
                                        index=0, disabled=(algo_clf not in ("SVM", "Comparar todos")))
            lr_xgb_v = p6.slider("learning_rate (XGBoost)", 0.01, 1.0, 0.1, 0.01)

        if st.button("Entrenar modelo(s)", type="primary", key="btn_clf"):
            df_work = df.copy()
            y_raw = df_work[target_clf]

            le = None
            if not pd.api.types.is_numeric_dtype(y_raw):
                le = LabelEncoder()
                df_work[target_clf] = le.fit_transform(y_raw.astype(str))

            maestro_clf = AgenteMaestro(df_work)
            params = {
                "n_neighbors": n_neighbors_v,
                "max_depth": max_depth_v,
                "n_estimators": n_est_v,
                "min_samples_split": 2,
                "C": c_reg_v,
                "kernel": kernel_svm_v,
                "learning_rate": lr_xgb_v,
            }

            try:
                with st.spinner("Entrenando..."):
                    if algo_clf == "Comparar todos":
                        resultados, tabla = maestro_clf.comparar_clasificacion(
                            target=target_clf, params=params,
                            test_size=test_size / 100, random_state=int(random_seed)
                        )
                        best = max(resultados, key=lambda r: r["exactitud"])
                    else:
                        res = maestro_clf.ejecutar_clasificacion(
                            target=target_clf, algoritmo=algo_clf, params=params,
                            test_size=test_size / 100, random_state=int(random_seed)
                        )
                        resultados = [res]
                        roc_val_single = res.get("roc_auc")
                        tabla = pd.DataFrame([{
                            "Modelo": res["modelo_nombre"],
                            "Exactitud": res["exactitud"],
                            "F1-Score": res["f1_ponderado"],
                            "ROC-AUC": roc_val_single if roc_val_single is not None else "N/A",
                            "Error Global": res["error_global"],
                        }])
                        best = res
            except Exception as e:
                st.error(f"Error entrenando: {e}")
                st.stop()

            clases_labels = (le.classes_.astype(str)
                             if le is not None else np.array([str(c) for c in best["clases"]]))

            st.markdown("---")
            st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)

            kb1, kb2, kb3, kb4, kb5 = st.columns(5)
            kb1.markdown(metric_card("Mejor Modelo", best["modelo_nombre"], ""), unsafe_allow_html=True)
            kb2.markdown(metric_card("Exactitud", fmt_pct(best["exactitud"]),
                                     "en datos de test", score_color(best["exactitud"])),
                         unsafe_allow_html=True)
            kb3.markdown(metric_card("F1-Score", f"{best['f1_ponderado']:.3f}",
                                     "ponderado", score_color(best["f1_ponderado"])),
                         unsafe_allow_html=True)
            roc_val = best.get("roc_auc")
            if roc_val is not None:
                kb4.markdown(metric_card("ROC-AUC", f"{roc_val:.3f}",
                                         "ponderado OvR", score_color(roc_val)),
                             unsafe_allow_html=True)
            else:
                kb4.markdown(metric_card("ROC-AUC", "N/A",
                                         "no soportado", "yellow"),
                             unsafe_allow_html=True)
            kb5.markdown(metric_card("Muestras test", len(best["y_test"]),
                                     "", "yellow"), unsafe_allow_html=True)

            if len(resultados) > 1:
                st.markdown("---")
                st.markdown("**Comparación de modelos**")
                cg1, cg2 = st.columns(2)
                with cg1:
                    st.dataframe(
                        tabla.style
                            .format({"Exactitud": "{:.2%}", "F1-Score": "{:.3f}",
                                     "ROC-AUC": "{:.3f}", "Error Global": "{:.3f}"})
                            .background_gradient(subset=["Exactitud", "F1-Score", "ROC-AUC"], cmap="Greens"),
                        width='stretch', hide_index=True)
                with cg2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(tabla["Modelo"], tabla["Exactitud"],
                            color=sns.color_palette("Set2", len(tabla)),
                            edgecolor="white", linewidth=0.5)
                    for i, val in enumerate(tabla["Exactitud"]):
                        ax.text(val + 0.005, i, f"{val:.1%}", va="center", fontsize=9)
                    ax.set_xlim(0, 1.12)
                    ax.set_xlabel("Exactitud")
                    ax.set_title("Exactitud por modelo")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    show_fig(fig)

            st.markdown("---")
            st.markdown(f"**Matriz de Confusión - {best['modelo_nombre']}**")
            cm = best["matriz_confusion"]
            fig, ax = plt.subplots(figsize=(max(5, len(clases_labels) * 1.5 + 2),
                                            max(4, len(clases_labels) * 1.5 + 1.5)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=clases_labels, yticklabels=clases_labels,
                        linewidths=0.5, linecolor="white", ax=ax)
            ax.set_xlabel("Predicho", fontsize=11)
            ax.set_ylabel("Real", fontsize=11)
            ax.set_title(f"Matriz de Confusión - {best['modelo_nombre']}", fontsize=12)
            plt.tight_layout()
            show_fig(fig)

            st.markdown("---")
            st.markdown(f"**Reporte de clasificación - {best['modelo_nombre']}**")
            report_df = pd.DataFrame(best["reporte"]).T.drop(columns=["support"], errors="ignore")
            st.dataframe(
                report_df.style.format(precision=3)
                    .background_gradient(subset=["precision", "recall", "f1-score"], cmap="Greens"),
                width='stretch')


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 · REGRESIÓN
# ════════════════════════════════════════════════════════════════════════════
with tab_reg:
    df = st.session_state.df
    dataset_cargado = df is not None
    if not dataset_cargado:
        st.info(" Sube un archivo CSV desde la barra lateral para ejecutar regresión.")
    else:
        sugerencia = st.session_state.sugerencia
        st.markdown('<div class="section-title">Configuración</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            num_options = df.select_dtypes(include="number").columns.tolist()
            _reg_opts = [None] + num_options
            _reg_default = (_reg_opts.index(sugerencia["target_sugerido"])
                            if sugerencia is not None and sugerencia["tipo"] == "regresion"
                            and sugerencia["target_sugerido"] in num_options else 0)
            target_reg = st.selectbox("Variable objetivo (numérica)",
                                      _reg_opts, index=_reg_default, key="tgt_reg",
                                      help="Solo columnas numéricas continuas.")
        with rc2:
            algo_reg = st.selectbox("Algoritmo", [
                "Regresión Lineal Múltiple", "Lasso", "LassoCV", "Ridge", "RidgeCV",
                "SVM", "Árbol de Decisión", "Random Forest",
                "XGBoost", "Gradient Boosting", "Comparar todos"
            ], key="alg_reg")

        if target_reg is None:
            st.info("Selecciona la variable objetivo para continuar.")

        with st.expander("Parámetros del modelo"):
            rp1, rp2, rp3 = st.columns(3)
            max_depth_r = rp1.slider("Max depth (árboles)", 1, 15, 4, key="md_reg")
            n_est_r = rp2.slider("N estimadores (ensembles)", 10, 500, 100, 10, key="ne_reg")
            lr_reg = rp3.slider("learning_rate (XGBoost)", 0.01, 1.0, 0.1, 0.01, key="lr_reg")

        if st.button("Entrenar modelo(s)", type="primary", key="btn_reg"):
            df_reg_input = df.select_dtypes(include="number").dropna()
            if target_reg not in df_reg_input.columns:
                st.error(f"La columna '{target_reg}' no es numérica o tiene nulos.")
                st.stop()

            maestro_reg = AgenteMaestro(df_reg_input)
            params = {"max_depth": max_depth_r, "n_estimators": n_est_r, "learning_rate": lr_reg}

            try:
                with st.spinner("Entrenando modelos..."):
                    if algo_reg == "Comparar todos":
                        resultados, tabla, errores = maestro_reg.comparar_regresion(
                            target=target_reg, params=params,
                            test_size=test_size / 100, random_state=int(random_seed)
                        )
                        if errores:
                            with st.expander(f"{len(errores)} algoritmo(s) fallaron", expanded=False):
                                for err in errores:
                                    st.error(f"**{err['algoritmo']}** - {err['tipo']}: {err['mensaje']}")
                                    st.code(err["traceback"], language="python")
                        if not resultados:
                            st.error("Ningún modelo de regresión pudo entrenarse. Revisa los errores arriba.")
                            st.stop()
                        best_r = max(resultados, key=lambda r: r["r2"])
                    else:
                        res = maestro_reg.ejecutar_regresion(
                            target=target_reg, algoritmo=algo_reg, params=params,
                            test_size=test_size / 100, random_state=int(random_seed)
                        )
                        resultados = [res]
                        mape_single = res.get("mape")
                        tabla = pd.DataFrame([{
                            "Modelo": res["modelo_nombre"],
                            "R²": res["r2"],
                            "MSE": res["mse"],
                            "RMSE": res["rmse"],
                            "MAE": res["mae"],
                            "MAPE": mape_single if not (isinstance(mape_single, float) and np.isnan(mape_single)) else "N/A",
                        }])
                        best_r = res
            except Exception as e:
                st.error(f"Error entrenando: {e}")
                st.stop()

            st.markdown("---")
            st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)

            rb1, rb2, rb3, rb4, rb5, rb6 = st.columns(6)
            rb1.markdown(metric_card("Mejor Modelo", best_r["modelo_nombre"], ""), unsafe_allow_html=True)
            rb2.markdown(metric_card("R²", f"{best_r['r2']:.4f}", "Varianza explicada",
                                     score_color(max(0, best_r["r2"]))), unsafe_allow_html=True)
            rb3.markdown(metric_card("MSE", f"{best_r['mse']:.4f}", "Error cuadrático", "red"),
                         unsafe_allow_html=True)
            rb4.markdown(metric_card("RMSE", f"{best_r['rmse']:.4f}", "Raíz error cuadrático", "red"),
                         unsafe_allow_html=True)
            rb5.markdown(metric_card("MAE", f"{best_r['mae']:.4f}", "Error absoluto medio", "yellow"),
                         unsafe_allow_html=True)
            mape_val = best_r.get("mape")
            if mape_val is not None and not (isinstance(mape_val, float) and np.isnan(mape_val)):
                rb6.markdown(metric_card("MAPE", f"{mape_val:.2f}%", "Error porcentual", "yellow"),
                             unsafe_allow_html=True)
            else:
                rb6.markdown(metric_card("MAPE", "N/A", "no disponible", "yellow"),
                             unsafe_allow_html=True)

            if len(resultados) > 1:
                rg1, rg2 = st.columns(2)
                with rg1:
                    st.markdown("**Tabla comparativa**")
                    st.dataframe(
                        tabla.style
                            .format({"MSE": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}%", "R²": "{:.4f}"})
                            .background_gradient(subset=["R²"], cmap="Greens")
                            .background_gradient(subset=["MSE", "RMSE", "MAE", "MAPE"], cmap="Reds_r"),
                        width='stretch', hide_index=True)
                with rg2:
                    st.markdown("**R² por modelo**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ["#00c49a" if m == best_r["modelo_nombre"] else "#4f8bf9"
                              for m in tabla["Modelo"]]
                    ax.barh(tabla["Modelo"], tabla["R²"], color=colors,
                            edgecolor="white", linewidth=0.5)
                    ax.set_xlabel("R²")
                    ax.set_title("R² por modelo")
                    ax.invert_yaxis()
                    ax.set_xlim(min(0, tabla["R²"].min() - 0.05), 1.1)
                    for i, val in enumerate(tabla["R²"]):
                        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)
                    plt.tight_layout()
                    show_fig(fig)

            st.markdown("---")
            st.markdown(f"**Predicho vs Real - {best_r['modelo_nombre']}**")
            y_test_arr = np.asarray(best_r["y_test"])
            preds_b = np.asarray(best_r["y_pred"])
            n_sample = min(len(y_test_arr), 200)
            idx = np.random.RandomState(int(random_seed)).choice(len(y_test_arr), n_sample, replace=False)

            fg1, fg2 = st.columns(2)
            with fg1:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(y_test_arr[idx], preds_b[idx], alpha=0.6, color="#4f8bf9",
                           edgecolors="white", linewidths=0.3, s=40)
                mn = float(min(y_test_arr[idx].min(), preds_b[idx].min()))
                mx = float(max(y_test_arr[idx].max(), preds_b[idx].max()))
                ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Línea perfecta")
                ax.set_xlabel("Valores reales")
                ax.set_ylabel("Valores predichos")
                ax.set_title("Real vs. Predicho")
                ax.legend()
                plt.tight_layout()
                show_fig(fig)
            with fg2:
                residuals = y_test_arr[idx] - preds_b[idx]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.hist(residuals, bins=30, color="#4f8bf9", edgecolor="white", alpha=0.8)
                ax.axvline(0, color="#ff4b6e", linestyle="--", linewidth=1.5)
                ax.set_xlabel("Residuo (Real − Predicho)")
                ax.set_ylabel("Frecuencia")
                ax.set_title("Distribución de residuos")
                plt.tight_layout()
                show_fig(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 · DASHBOARD EJECUTIVO
# ════════════════════════════════════════════════════════════════════════════
with tab_dash:
    _df = st.session_state.df
    if _df is None:
        st.info(" Sube un archivo CSV desde la pestaña **Carga de Datos** para generar el dashboard ejecutivo.")
    else:
        _sg = st.session_state.sugerencia
        _nulos = _df.isna().sum().sum()
        _total = _df.size
        _pct_nulos = _nulos / _total * 100 if _total else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(metric_card("Filas", f"{_df.shape[0]:,}", "registros"), unsafe_allow_html=True)
        k2.markdown(metric_card("Columnas", str(_df.shape[1]), "variables"), unsafe_allow_html=True)
        k3.markdown(metric_card("Nulos", f"{_pct_nulos:.1f}%", f"{_nulos:,} celdas", "yellow" if _pct_nulos > 0 else "green"), unsafe_allow_html=True)
        tipo = _sg["tipo"].capitalize() if _sg and _sg.get("tipo") else "—"
        k4.markdown(metric_card("Problema", tipo, ""), unsafe_allow_html=True)
        target = _sg["target_sugerido"] if _sg and _sg.get("target_sugerido") else "—"
        k5.markdown(metric_card("Target", target, ""), unsafe_allow_html=True)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            num = _df.select_dtypes(include="number")
            if num.shape[1] >= 2:
                st.markdown("**Correlaciones**")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                            center=0, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)
        with c2:
            nulls = _df.isna().sum()
            nulls = nulls[nulls > 0]
            if not nulls.empty:
                st.markdown("**Valores nulos por columna**")
                fig, ax = plt.subplots(figsize=(5, 3))
                nulls.sort_values().plot.barh(ax=ax, color="#e74c3c", width=0.7)
                ax.set_xlabel("Cantidad")
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)
            else:
                st.markdown("**Valores nulos**")
                st.success("No hay valores nulos en el dataset.")

        st.divider()
        if st.button("Ejecutar benchmark completo", key="btn_dash"):
            with st.spinner("Ejecutando benchmark..."):
                resultados = {}
                if _sg and _sg.get("tipo") == "clasificacion":
                    tgt = _sg["target_sugerido"]
                    if tgt and tgt in _df.columns:
                        w = _df.dropna().copy()
                        y = w[tgt]
                        if not pd.api.types.is_numeric_dtype(y):
                            le = LabelEncoder()
                            w[tgt] = le.fit_transform(y.astype(str))
                        r, t = AgenteMaestro(w).comparar_clasificacion(
                            target=tgt, test_size=test_size/100, random_state=int(random_seed))
                        resultados["clasificacion"] = {"tabla": t,
                            "best": max(r, key=lambda x: x["exactitud"])}
                if _sg and _sg.get("tipo") == "regresion":
                    tgt = _sg["target_sugerido"]
                    if tgt and tgt in _df.select_dtypes(include="number").columns:
                        n = _df.select_dtypes(include="number").dropna()
                        r, t, _ = AgenteMaestro(n).comparar_regresion(
                            target=tgt, test_size=test_size/100, random_state=int(random_seed))
                        resultados["regresion"] = {"tabla": t,
                            "best": max(r, key=lambda x: x["r2"])}
                n = _df.select_dtypes(include="number").dropna()
                if len(n.columns) >= 2:
                    m = AgenteMaestro(n)
                    r, t = m.comparar_clustering(
                        n_clusters=3, random_state=int(random_seed))
                    resultados["clustering"] = {"tabla": t,
                        "best": m.decidir_clustering(r)}
            if not resultados:
                st.info("No se ejecutó ningún benchmark.")
            for area, d in resultados.items():
                st.markdown(f"**{area.capitalize()}**")
                st.dataframe(d["tabla"].style.format(precision=4).background_gradient(cmap="Greens"),
                             width='stretch', hide_index=True)
                b = d["best"]
                st.success(f"Mejor: {b.get('modelo_nombre') or b.get('algoritmo','?')}")
            if resultados:
                st.divider()
                st.markdown("**Conclusiones automáticas**")
                for area, d in resultados.items():
                    b = d["best"]
                    nombre = b.get("modelo_nombre") or b.get("algoritmo", "?")
                    if area == "clasificacion":
                        acc = b.get("exactitud", 0)
                        f1 = b.get("f1_ponderado", 0)
                        level = "★ Bueno" if acc >= 0.8 else "★ Aceptable" if acc >= 0.6 else "★ Bajo"
                        st.markdown(f"- **{area.capitalize()}**: {nombre} lidera con exactitud {acc:.1%} y F1 {f1:.1%} — rendimiento **{level}**.")
                    elif area == "regresion":
                        r2 = b.get("r2", 0)
                        level = "★ Bueno" if r2 >= 0.7 else "★ Aceptable" if r2 >= 0.4 else "★ Bajo"
                        st.markdown(f"- **{area.capitalize()}**: {nombre} lidera con R² {r2:.3f} — poder predictivo **{level}**.")
                    elif area == "clustering":
                        sil = b.get("silhouette", b.get("metricas", {}).get("silhouette", 0))
                        level = "★ Buena separación" if sil >= 0.5 else "★ Estructura débil" if sil >= 0.25 else "★ Poca estructura"
                        st.markdown(f"- **{area.capitalize()}**: {nombre} obtiene Silhouette {sil:.3f} — {level}.")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ML Framework · Ciclo 1 — 2026 · Inteligencia Artificial")
