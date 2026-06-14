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

from eda import analisisEDA
from clasificacion import Clasificacion
from regresion import Regresion
from clustering import Clustering

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Framework",
    page_icon="🤖",
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


# ── Sidebar: carga de datos ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Framework")
    st.markdown("---")
    st.markdown("### 📂 Cargar datos")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"], label_visibility="collapsed")
    sep = st.radio("Separador", [",", ";"], horizontal=True)
    use_dummies = st.checkbox("Convertir categóricas a dummies", value=False,
                              help="Aplica get_dummies a las columnas categóricas")

    st.markdown("---")
    st.markdown("### ⚙️ Configuración global")
    test_size = st.slider("Tamaño de test (%)", 10, 40, 25)
    random_seed = st.number_input("Semilla aleatoria", value=42, step=1)

    st.markdown("---")
    st.caption("Ciclo 1 — 2026 · IA · MLFramework")


# ── Cargar dataset ───────────────────────────────────────────────────────────
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        if use_dummies:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                df = df.astype({c: int for c in df.select_dtypes(include='bool').columns})
    except Exception as e:
        st.sidebar.error(f"Error al leer el CSV: {e}")


# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(90deg,#1a1f35,#1e3a5f);
            border-radius:14px; padding:28px 32px; margin-bottom:24px;">
    <h1 style="margin:0;color:#f0f2f6;font-size:32px;">🤖 ML Framework</h1>
    <p style="margin:6px 0 0;color:#9ca3af;font-size:15px;">
        Análisis exploratorio · Clustering · Clasificación · Regresión
    </p>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.info("👈 Sube un archivo CSV desde la barra lateral para comenzar.")
    st.stop()


# ── Instanciar EDA (compartido por las tabs) ─────────────────────────────────
eda = analisisEDA(None, None)
eda.df = df


# ── Sugerencia automática de tipo de problema ────────────────────────────────
sugerencia = analisisEDA.detectar_tipo_problema(df)
_iconos = {"clasificacion": "🎯", "regresion": "📈", "no_supervisado": "🔵"}
_etiquetas = {"clasificacion": "Clasificación", "regresion": "Regresión",
              "no_supervisado": "No supervisado (Clustering / EDA)"}
_icono = _iconos.get(sugerencia["tipo"], "🤖")
_etiqueta = _etiquetas.get(sugerencia["tipo"], sugerencia["tipo"])

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧠 Sugerencia automática")
    st.markdown(f"**{_icono} {_etiqueta}**")
    if sugerencia["target_sugerido"]:
        st.caption(f"Target sugerido: `{sugerencia['target_sugerido']}`")
    st.caption(sugerencia["razon"])

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


# ── Tabs principales (agrupadas por tipo de aprendizaje) ─────────────────────
tab_eda, tab_sup, tab_ns = st.tabs([
    "📊 EDA",
    "🧠 Supervisado",
    "🔵 No supervisado",
])

with tab_sup:
    st.caption("Modelos que requieren una variable objetivo (target).")
    tab_clf, tab_reg = st.tabs([
        "🎯 Clasificación",
        "📈 Regresión",
    ])

with tab_ns:
    st.caption("Modelos que no requieren etiquetas — descubren estructura en los datos.")
    (tab_cluster,) = st.tabs([
        "🔵 Clustering",
    ])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-title">Resumen del Dataset</div>', unsafe_allow_html=True)

    resumen = eda.resumen_dict()
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
        st.dataframe(df.head(8), use_container_width=True)
    with col_types:
        st.markdown("**Tipos de datos**")
        dtype_df = pd.DataFrame({"Columna": df.columns,
                                 "Tipo": df.dtypes.astype(str).values,
                                 "Nulos": df.isnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Estadísticas Descriptivas</div>', unsafe_allow_html=True)
    st.dataframe(resumen["descripcion"].T.style.format(precision=3), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Visualizaciones</div>', unsafe_allow_html=True)

    viz_type = st.selectbox("Tipo de gráfico", [
        "Boxplots", "Histogramas", "Distribución (KDE)", "Densidad",
        "Correlaciones (heatmap)", "Dispersión por pares"
    ])

    if num_cols:
        eda_viz = analisisEDA(None, None)
        eda_viz.df = df.select_dtypes(include="number").dropna()
        try:
            if viz_type == "Boxplots":
                show_fig(eda_viz.graficoBoxplot_fig())
            elif viz_type == "Histogramas":
                show_fig(eda_viz.histogramas_fig())
            elif viz_type == "Distribución (KDE)":
                show_fig(eda_viz.distribucionVariables_fig())
            elif viz_type == "Densidad":
                show_fig(eda_viz.datosDensidad_fig())
            elif viz_type == "Correlaciones (heatmap)":
                show_fig(eda_viz.graficoCorrelacion_fig())
            elif viz_type == "Dispersión por pares":
                if len(num_cols) <= 6:
                    fig = eda_viz.graficosDispersion_fig()
                    if fig is not None:
                        show_fig(fig)
                else:
                    sel = st.multiselect("Selecciona hasta 5 columnas", num_cols, num_cols[:5])
                    if len(sel) >= 2:
                        fig = eda_viz.graficosDispersion_fig(columnas=sel)
                        if fig is not None:
                            show_fig(fig)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")

    if cat_cols:
        st.markdown("---")
        st.markdown('<div class="section-title">Columnas Categóricas</div>', unsafe_allow_html=True)
        col_sel = st.selectbox("Ver distribución de", cat_cols)
        vc = df[col_sel].value_counts().reset_index()
        vc.columns = [col_sel, "Frecuencia"]
        vc["Porcentaje"] = (vc["Frecuencia"] / vc["Frecuencia"].sum() * 100).round(1).astype(str) + "%"

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.dataframe(vc, use_container_width=True, hide_index=True)
        with cc2:
            eda_cat = analisisEDA(None, None)
            eda_cat.df = df
            show_fig(eda_cat.histogramaClase_fig(col_sel))


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · CLUSTERING (usa Clustering del framework)
# ════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown('<div class="section-title">Configuración del Clustering</div>', unsafe_allow_html=True)

    df_num_cl = df.select_dtypes(include="number").dropna()
    if df_num_cl.empty or df_num_cl.shape[1] < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas para clustering.")
        st.stop()

    cl1, cl2 = st.columns([1, 2])
    with cl1:
        algos_disponibles = ["KMeans", "KMedoids", "HAC", "T-SNE", "UMAP"]
        algo_cl = st.selectbox("Algoritmo", algos_disponibles)
        n_clusters_cl = st.slider("Número de clusters", 2, 10, 3)
        normalizar = st.checkbox("Normalizar (StandardScaler)", value=True)
        if algo_cl == "HAC":
            linkage_method = st.selectbox("Método de linkage", ["ward", "average", "single", "complete"])
        else:
            linkage_method = "ward"
    with cl2:
        st.markdown("**Columnas disponibles**")
        cols_sel = st.multiselect("Usar columnas", df_num_cl.columns.tolist(),
                                  df_num_cl.columns.tolist())

    if not cols_sel or len(cols_sel) < 2:
        st.warning("Selecciona al menos 2 columnas.")
        st.stop()

    if st.button("▶️ Ejecutar clustering", type="primary"):
        try:
            clu = Clustering(df)
            with st.spinner(f"Entrenando {algo_cl}..."):
                kwargs = {"linkage": linkage_method} if algo_cl == "HAC" else {}
                res = clu.ejecutar(
                    algoritmo=algo_cl,
                    n_clusters=n_clusters_cl,
                    columnas=cols_sel,
                    normalizar=normalizar,
                    random_state=int(random_seed),
                    **kwargs,
                )
        except Exception as e:
            st.error(f"Error ejecutando clustering: {e}")
            st.stop()

        mtr = res["metricas"]
        st.markdown("---")
        st.markdown('<div class="section-title">Métricas del Modelo</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Algoritmo", res["algoritmo"], ""), unsafe_allow_html=True)
        m2.markdown(metric_card("Silhouette", f"{mtr['silhouette']:.3f}",
                                "Rango: -1 → 1", score_color(mtr["silhouette"])),
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
            fig = Clustering.plot_proyeccion_2d_fig(
                res["X_proj"], res["labels"], n_clusters_cl,
                titulo=f"{algo_cl} — {n_clusters_cl} clusters"
            )
            show_fig(fig)
        with g2:
            st.markdown("**Distribución por cluster**")
            fig = Clustering.plot_distribucion_clusters_fig(res["labels"], n_clusters_cl)
            show_fig(fig)

        st.markdown("---")
        st.markdown("**Perfil de cada cluster (media de variables)**")
        perfil = clu.perfil_clusters(res)
        st.dataframe(
            perfil.style.background_gradient(cmap="Blues", axis=0).format(precision=3),
            use_container_width=True
        )

        if algo_cl == "HAC" and len(res["X"]) <= 200:
            st.markdown("---")
            st.markdown(f"**Dendrograma — método: {linkage_method}**")
            fig = Clustering.plot_dendrograma_fig(res["X"], metodo=linkage_method)
            show_fig(fig)

        if len(res["X"]) <= 2000:
            st.markdown("---")
            st.markdown("**Análisis del Silhouette Score por número de clusters**")
            X_for_curve = res["X_proj"] if algo_cl in ("T-SNE", "UMAP") else res["X"]
            fig, best_k, best_score = Clustering.plot_silhouette_curve_fig(
                X_for_curve, k_min=2, k_max=10, random_state=int(random_seed)
            )
            show_fig(fig)
            st.info(f"Mejor k según Silhouette: **k = {best_k}** (score = {best_score:.3f})")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · CLASIFICACIÓN (usa Clasificacion del framework)
# ════════════════════════════════════════════════════════════════════════════
with tab_clf:
    st.markdown('<div class="section-title">Configuración</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        _clf_opts = [None] + list(df.columns)
        _clf_default = (_clf_opts.index(sugerencia["target_sugerido"])
                        if sugerencia["tipo"] == "clasificacion"
                        and sugerencia["target_sugerido"] in df.columns else 0)
        target_clf = st.selectbox("Columna objetivo (target)", _clf_opts,
                                  index=_clf_default, key="tgt_clf",
                                  help="Sugerencia automática aplicada si el dataset parece de clasificación.")
    with cc2:
        algo_clf = st.selectbox("Algoritmo", [
            "KNN", "Decision Tree", "Random Forest",
            "Gradient Boosting", "AdaBoost", "Comparar todos"
        ], key="alg_clf")

    if target_clf is None:
        st.info("Selecciona la columna target para continuar.")
        st.stop()

    with st.expander("⚙️ Parámetros del modelo"):
        p1, p2, p3 = st.columns(3)
        n_neighbors_v = p1.slider("KNN — vecinos", 1, 20, 5)
        max_depth_v = p2.slider("Max depth", 1, 15, 4)
        n_est_v = p3.slider("N estimadores", 10, 300, 100, 10)

    if st.button("▶️ Entrenar modelo(s)", type="primary", key="btn_clf"):
        df_work = df.copy()
        y_raw = df_work[target_clf]

        # Encode target si no es numérico
        le = None
        if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
            le = LabelEncoder()
            df_work[target_clf] = le.fit_transform(y_raw.astype(str))

        clf = Clasificacion(df_work)
        params = {
            "n_neighbors": n_neighbors_v,
            "max_depth": max_depth_v,
            "n_estimators": n_est_v,
            "min_samples_split": 2,
        }

        try:
            with st.spinner("Entrenando..."):
                if algo_clf == "Comparar todos":
                    resultados, tabla = clf.comparar_todos(
                        target=target_clf, params=params,
                        test_size=test_size / 100, random_state=int(random_seed)
                    )
                    best = max(resultados, key=lambda r: r["exactitud"])
                else:
                    res = clf.entrenar(
                        algo_clf, target=target_clf, params=params,
                        test_size=test_size / 100, random_state=int(random_seed)
                    )
                    resultados = [res]
                    tabla = pd.DataFrame([{
                        "Modelo": res["modelo_nombre"],
                        "Exactitud": res["exactitud"],
                        "F1-Score": res["f1_ponderado"],
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

        kb1, kb2, kb3, kb4 = st.columns(4)
        kb1.markdown(metric_card("Mejor Modelo", best["modelo_nombre"], ""), unsafe_allow_html=True)
        kb2.markdown(metric_card("Exactitud", fmt_pct(best["exactitud"]),
                                 "en datos de test", score_color(best["exactitud"])),
                     unsafe_allow_html=True)
        kb3.markdown(metric_card("F1-Score", f"{best['f1_ponderado']:.3f}",
                                 "ponderado", score_color(best["f1_ponderado"])),
                     unsafe_allow_html=True)
        kb4.markdown(metric_card("Muestras test", len(best["y_test"]),
                                 "", "yellow"), unsafe_allow_html=True)

        if len(resultados) > 1:
            st.markdown("---")
            st.markdown("**Comparación de modelos**")
            cg1, cg2 = st.columns(2)
            with cg1:
                st.dataframe(
                    tabla.style
                        .format({"Exactitud": "{:.2%}", "F1-Score": "{:.3f}",
                                 "Error Global": "{:.3f}"})
                        .background_gradient(subset=["Exactitud", "F1-Score"], cmap="Greens"),
                    use_container_width=True, hide_index=True)
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
        st.markdown(f"**Matriz de Confusión — {best['modelo_nombre']}**")
        cm = best["matriz_confusion"]
        fig, ax = plt.subplots(figsize=(max(5, len(clases_labels) * 1.5 + 2),
                                        max(4, len(clases_labels) * 1.5 + 1.5)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=clases_labels, yticklabels=clases_labels,
                    linewidths=0.5, linecolor="white", ax=ax)
        ax.set_xlabel("Predicho", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        ax.set_title(f"Matriz de Confusión — {best['modelo_nombre']}", fontsize=12)
        plt.tight_layout()
        show_fig(fig)

        st.markdown("---")
        st.markdown(f"**Reporte de clasificación — {best['modelo_nombre']}**")
        report_df = pd.DataFrame(best["reporte"]).T.drop(columns=["support"], errors="ignore")
        st.dataframe(
            report_df.style.format(precision=3)
                .background_gradient(subset=["precision", "recall", "f1-score"], cmap="Greens"),
            use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 · REGRESIÓN (usa Regresion del framework)
# ════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown('<div class="section-title">Configuración</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        num_options = df.select_dtypes(include="number").columns.tolist()
        _reg_opts = [None] + num_options
        _reg_default = (_reg_opts.index(sugerencia["target_sugerido"])
                        if sugerencia["tipo"] == "regresion"
                        and sugerencia["target_sugerido"] in num_options else 0)
        target_reg = st.selectbox("Variable objetivo (numérica)",
                                  _reg_opts, index=_reg_default, key="tgt_reg",
                                  help="Sugerencia automática aplicada si el dataset parece de regresión.")
    with rc2:
        algo_reg = st.selectbox("Algoritmo", [
            "Regresión Lineal Múltiple", "Lasso", "Ridge",
            "SVM", "Árbol de Decisión", "Random Forest",
            "Gradient Boosting", "Comparar todos"
        ], key="alg_reg")

    if target_reg is None:
        st.info("Selecciona la variable objetivo para continuar.")
        st.stop()

    with st.expander("⚙️ Parámetros del modelo"):
        rp1, rp2 = st.columns(2)
        max_depth_r = rp1.slider("Max depth (árboles)", 1, 15, 4, key="md_reg")
        n_est_r = rp2.slider("N estimadores (ensembles)", 10, 500, 100, 10, key="ne_reg")

    if st.button("▶️ Entrenar modelo(s)", type="primary", key="btn_reg"):
        df_reg_input = df.select_dtypes(include="number").dropna()
        if target_reg not in df_reg_input.columns:
            st.error(f"La columna '{target_reg}' no es numérica o tiene nulos.")
            st.stop()

        reg = Regresion(df_reg_input)
        params = {"max_depth": max_depth_r, "n_estimators": n_est_r}

        try:
            with st.spinner("Entrenando modelos..."):
                if algo_reg == "Comparar todos":
                    resultados, tabla, errores = reg.comparar_todos(
                        target=target_reg, params=params,
                        test_size=test_size / 100, random_state=int(random_seed)
                    )
                    if errores:
                        with st.expander(f"⚠️ {len(errores)} algoritmo(s) fallaron", expanded=False):
                            for err in errores:
                                st.error(f"**{err['algoritmo']}** — {err['tipo']}: {err['mensaje']}")
                                st.code(err["traceback"], language="python")
                    if not resultados:
                        st.error("Ningún modelo de regresión pudo entrenarse. Revisa los errores arriba.")
                        st.stop()
                    best_r = max(resultados, key=lambda r: r["r2"])
                else:
                    res = reg.entrenar(
                        algo_reg, target=target_reg, params=params,
                        test_size=test_size / 100, random_state=int(random_seed)
                    )
                    resultados = [res]
                    tabla = pd.DataFrame([{
                        "Modelo": res["modelo_nombre"],
                        "R²": res["r2"],
                        "RMSE": res["rmse"],
                        "MAE": res["mae"],
                    }])
                    best_r = res
        except Exception as e:
            st.error(f"Error entrenando: {e}")
            st.stop()

        st.markdown("---")
        st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)

        rb1, rb2, rb3, rb4 = st.columns(4)
        rb1.markdown(metric_card("Mejor Modelo", best_r["modelo_nombre"], ""), unsafe_allow_html=True)
        rb2.markdown(metric_card("R²", f"{best_r['r2']:.4f}", "Varianza explicada",
                                 score_color(max(0, best_r["r2"]))), unsafe_allow_html=True)
        rb3.markdown(metric_card("RMSE", f"{best_r['rmse']:.4f}", "Error cuadrático medio", "red"),
                     unsafe_allow_html=True)
        rb4.markdown(metric_card("MAE", f"{best_r['mae']:.4f}", "Error absoluto medio", "yellow"),
                     unsafe_allow_html=True)

        if len(resultados) > 1:
            rg1, rg2 = st.columns(2)
            with rg1:
                st.markdown("**Tabla comparativa**")
                st.dataframe(
                    tabla.style
                        .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"})
                        .background_gradient(subset=["R²"], cmap="Greens")
                        .background_gradient(subset=["RMSE", "MAE"], cmap="Reds_r"),
                    use_container_width=True, hide_index=True)
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
        st.markdown(f"**Predicho vs Real — {best_r['modelo_nombre']}**")
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


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ML Framework · Ciclo 1 — 2026 · Inteligencia Artificial")
