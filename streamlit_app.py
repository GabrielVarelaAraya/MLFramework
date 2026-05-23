import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
warnings.filterwarnings('ignore')

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
    if score >= 0.75: return "green"
    if score >= 0.5:  return "yellow"
    return "red"

def fmt_pct(v):
    return f"{v*100:.1f}%"


# ── Sidebar: carga de datos ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Framework")
    st.markdown("---")
    st.markdown("### 📂 Cargar datos")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"], label_visibility="collapsed")
    sep = st.radio("Separador", [",", ";"], horizontal=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuración global")
    test_size   = st.slider("Tamaño de test (%)", 10, 40, 25)
    random_seed = st.number_input("Semilla aleatoria", value=42, step=1)

    st.markdown("---")
    st.caption("Ciclo 1 — 2026 · IA · MLFramework")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
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


# ── Tabs principales ─────────────────────────────────────────────────────────
tab_eda, tab_cluster, tab_clf, tab_reg = st.tabs([
    "📊 EDA",
    "🔵 Clustering",
    "🎯 Clasificación",
    "📈 Regresión",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-title">Resumen del Dataset</div>', unsafe_allow_html=True)

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object","category"]).columns.tolist()
    nulls_tot = int(df.isnull().sum().sum())
    dups      = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Filas", f"{df.shape[0]:,}", "registros totales"), unsafe_allow_html=True)
    c2.markdown(metric_card("Columnas", df.shape[1], f"{len(num_cols)} num · {len(cat_cols)} cat"), unsafe_allow_html=True)
    c3.markdown(metric_card("Valores nulos", nulls_tot, "en todo el dataset",
                            "red" if nulls_tot else "green"), unsafe_allow_html=True)
    c4.markdown(metric_card("Duplicados", dups, "filas duplicadas",
                            "red" if dups else "green"), unsafe_allow_html=True)

    st.markdown("---")

    col_prev, col_types = st.columns([2, 1])
    with col_prev:
        st.markdown("**Vista previa**")
        st.dataframe(df.head(8), use_container_width=True)
    with col_types:
        st.markdown("**Tipos de datos**")
        dtype_df = pd.DataFrame({"Columna": df.columns, "Tipo": df.dtypes.astype(str).values,
                                 "Nulos": df.isnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Estadísticas Descriptivas</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T.style.format(precision=3), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Visualizaciones</div>', unsafe_allow_html=True)

    viz_type = st.selectbox("Tipo de gráfico", [
        "Boxplots", "Histogramas", "Distribución (KDE)", "Densidad",
        "Correlaciones (heatmap)", "Dispersión por pares"
    ])

    if num_cols:
        from eda import analisisEDA
        eda_obj = analisisEDA(None, None)
        eda_obj.df = df.select_dtypes(include="number").dropna()
        try:
            if viz_type == "Boxplots":
                eda_obj.graficoBoxplot()
                st.pyplot(plt.gcf()); plt.clf()
            elif viz_type == "Histogramas":
                eda_obj.histogramas()
                st.pyplot(plt.gcf()); plt.clf()
            elif viz_type == "Distribución (KDE)":
                eda_obj.distribucionVariables()
                st.pyplot(plt.gcf()); plt.clf()
            elif viz_type == "Densidad":
                eda_obj.datosDensidad()
                st.pyplot(plt.gcf()); plt.clf()
            elif viz_type == "Correlaciones (heatmap)":
                eda_obj.graficoCorrelacion()
                st.pyplot(plt.gcf()); plt.clf()
            elif viz_type == "Dispersión por pares":
                if len(num_cols) <= 6:
                    eda_obj.graficosDispersion()
                    st.pyplot(plt.gcf()); plt.clf()
                else:
                    sel = st.multiselect("Selecciona hasta 5 columnas", num_cols, num_cols[:5])
                    if len(sel) >= 2:
                        eda_obj.df = df[sel].dropna()
                        eda_obj.graficosDispersion()
                        st.pyplot(plt.gcf()); plt.clf()
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
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = sns.color_palette("Set2", len(vc))
            ax.barh(vc[col_sel].astype(str), vc["Frecuencia"], color=colors)
            ax.set_xlabel("Frecuencia")
            ax.set_title(f"Distribución de '{col_sel}'")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig); plt.clf()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown('<div class="section-title">Configuración del Clustering</div>', unsafe_allow_html=True)

    df_num_cl = df.select_dtypes(include="number").dropna()
    if df_num_cl.empty:
        st.warning("No hay columnas numéricas suficientes para clustering.")
        st.stop()

    cl1, cl2 = st.columns([1, 2])
    with cl1:
        algo_cl = st.selectbox("Algoritmo", ["KMeans", "KMedoids", "HAC", "T-SNE"])
        n_clusters_cl = st.slider("Número de clusters", 2, 10, 3)
        normalizar = st.checkbox("Normalizar datos (StandardScaler)", value=True)
    with cl2:
        st.markdown("**Columnas disponibles**")
        cols_sel = st.multiselect("Usar columnas", df_num_cl.columns.tolist(),
                                  df_num_cl.columns.tolist())

    if not cols_sel:
        st.warning("Selecciona al menos 2 columnas.")
        st.stop()

    df_cl = df_num_cl[cols_sel]

    if st.button("▶️ Ejecutar clustering", type="primary"):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        X = StandardScaler().fit_transform(df_cl) if normalizar else df_cl.values
        labels = None

        with st.spinner(f"Entrenando {algo_cl}..."):
            if algo_cl == "KMeans":
                from sklearn.cluster import KMeans
                labels = KMeans(n_clusters=n_clusters_cl, random_state=random_seed,
                                n_init=10).fit_predict(X)
            elif algo_cl == "KMedoids":
                try:
                    from sklearn_extra.cluster import KMedoids
                    labels = KMedoids(n_clusters=n_clusters_cl, metric="cityblock",
                                      random_state=random_seed).fit_predict(X)
                except ImportError:
                    st.error("Instala sklearn-extra para usar KMedoids.")
                    st.stop()
            elif algo_cl == "HAC":
                from sklearn.cluster import AgglomerativeClustering
                labels = AgglomerativeClustering(n_clusters=n_clusters_cl,
                                                 linkage="ward").fit_predict(X)
            elif algo_cl == "T-SNE":
                from sklearn.manifold import TSNE
                from sklearn.cluster import KMeans
                X = TSNE(n_components=2, random_state=random_seed).fit_transform(X)
                labels = KMeans(n_clusters=n_clusters_cl, random_state=random_seed,
                                n_init=10).fit_predict(X)

        sil = silhouette_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        db  = davies_bouldin_score(X, labels)

        st.markdown("---")
        st.markdown('<div class="section-title">Métricas del Modelo</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Algoritmo", algo_cl, ""), unsafe_allow_html=True)
        m2.markdown(metric_card("Silhouette Score", f"{sil:.3f}",
                                "Rango: -1 (malo) → 1 (excelente)", score_color(sil)),
                    unsafe_allow_html=True)
        m3.markdown(metric_card("Calinski-Harabasz", f"{ch:.1f}",
                                "Cuanto mayor, mejor separación"), unsafe_allow_html=True)
        m4.markdown(metric_card("Davies-Bouldin", f"{db:.3f}",
                                "Cuanto menor, mejor compactación",
                                "green" if db < 1 else "yellow"), unsafe_allow_html=True)

        st.markdown("---")
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Proyección PCA 2D**")
            pca = PCA(n_components=2)
            comps = pca.fit_transform(X) if algo_cl != "T-SNE" else X
            var_exp = pca.explained_variance_ratio_ if algo_cl != "T-SNE" else [0, 0]
            fig, ax = plt.subplots(figsize=(6, 5))
            palette = sns.color_palette("Set2", n_clusters_cl)
            for k in range(n_clusters_cl):
                mask = labels == k
                ax.scatter(comps[mask, 0], comps[mask, 1], c=[palette[k]],
                           label=f"Cluster {k}", alpha=0.75,
                           edgecolors="white", linewidths=0.3, s=60)
            ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)" if algo_cl != "T-SNE" else "Dim 1")
            ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)" if algo_cl != "T-SNE" else "Dim 2")
            ax.set_title(f"{algo_cl} — {n_clusters_cl} clusters")
            ax.legend(framealpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.clf()

        with g2:
            st.markdown("**Distribución por cluster**")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar([f"Cluster {i}" for i in cluster_counts.index],
                          cluster_counts.values,
                          color=sns.color_palette("Set2", n_clusters_cl),
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, cluster_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(val), ha="center", va="bottom", fontweight="bold", color="white")
            ax.set_ylabel("Cantidad de puntos")
            ax.set_title("Tamaño de cada cluster")
            plt.tight_layout()
            st.pyplot(fig); plt.clf()

        st.markdown("---")
        st.markdown("**Perfil de cada cluster (media de variables)**")
        df_cl_labels = df_cl.copy()
        df_cl_labels["Cluster"] = labels
        perfil = df_cl_labels.groupby("Cluster").mean().round(3)
        st.dataframe(
            perfil.style.background_gradient(cmap="Blues", axis=0).format(precision=3),
            use_container_width=True
        )

        if len(df_cl) <= 2000:
            st.markdown("---")
            st.markdown("**Análisis del Silhouette Score por número de clusters**")
            from sklearn.metrics import silhouette_score as ss
            ks = range(2, min(11, len(df_cl)))
            scores = []
            for k in ks:
                from sklearn.cluster import KMeans as KM
                lbl = KM(n_clusters=k, random_state=random_seed, n_init=10).fit_predict(X)
                scores.append(ss(X, lbl))
            best_k = list(ks)[np.argmax(scores)]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(list(ks), scores, marker="o", color="#4f8bf9", linewidth=2.5, markersize=8)
            ax.fill_between(list(ks), scores, alpha=0.15, color="#4f8bf9")
            ax.axvline(best_k, color="#00c49a", linestyle="--", alpha=0.7,
                       label=f"Mejor k = {best_k}")
            ax.set_xlabel("Número de clusters (k)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score vs. Número de clusters")
            ax.legend(); ax.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig); plt.clf()
            st.info(f"El número óptimo de clusters según Silhouette Score es **k = {best_k}** "
                    f"(score = {max(scores):.3f})")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · CLASIFICACIÓN
# ════════════════════════════════════════════════════════════════════════════
with tab_clf:
    st.markdown('<div class="section-title">Configuración</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        target_clf = st.selectbox("Columna objetivo (target)", [None] + list(df.columns))
    with cc2:
        algo_clf = st.selectbox("Algoritmo", ["KNN", "Árbol de Decisión", "Random Forest",
                                               "Gradient Boosting", "AdaBoost", "Comparar todos"])

    if target_clf is None:
        st.info("Selecciona la columna target para continuar.")
        st.stop()

    with st.expander("⚙️ Parámetros del modelo"):
        p1, p2, p3 = st.columns(3)
        n_neighbors   = p1.slider("KNN — vecinos", 1, 20, 5)
        max_depth_clf = p2.slider("Max depth", 1, 15, 4)
        n_est_clf     = p3.slider("N estimadores", 10, 300, 100, 10)

    if st.button("▶️ Entrenar modelo(s)", type="primary", key="btn_clf"):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (confusion_matrix, classification_report,
                                     accuracy_score, f1_score)

        y_raw = df[target_clf]
        X_raw = df.drop(columns=[target_clf]).select_dtypes(include="number")
        X_raw = X_raw.dropna()
        y_raw = y_raw.loc[X_raw.index]

        if X_raw.empty:
            st.error("No hay columnas numéricas para entrenar.")
            st.stop()

        le    = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        X_sc  = StandardScaler().fit_transform(X_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X_sc, y_enc, test_size=test_size/100, random_state=random_seed)

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import (RandomForestClassifier,
                                      GradientBoostingClassifier, AdaBoostClassifier)

        model_map = {
            "KNN":               KNeighborsClassifier(n_neighbors=n_neighbors),
            "Árbol de Decisión": DecisionTreeClassifier(max_depth=max_depth_clf,
                                                         random_state=random_seed),
            "Random Forest":     RandomForestClassifier(n_estimators=n_est_clf,
                                                         max_depth=max_depth_clf,
                                                         random_state=random_seed),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_est_clf,
                                                              max_depth=max_depth_clf,
                                                              random_state=random_seed),
            "AdaBoost":          AdaBoostClassifier(n_estimators=n_est_clf,
                                                     random_state=random_seed),
        }
        to_run = model_map if algo_clf == "Comparar todos" else {algo_clf: model_map[algo_clf]}

        results = []
        with st.spinner("Entrenando..."):
            for name, mdl in to_run.items():
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)
                results.append({
                    "Modelo":    name,
                    "Exactitud": accuracy_score(y_test, preds),
                    "F1-Score":  f1_score(y_test, preds, average="weighted", zero_division=0),
                    "cm":        confusion_matrix(y_test, preds),
                    "preds":     preds,
                })

        best = max(results, key=lambda r: r["Exactitud"])
        st.markdown("---")
        st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)

        kb1, kb2, kb3, kb4 = st.columns(4)
        kb1.markdown(metric_card("Mejor Modelo", best["Modelo"], ""), unsafe_allow_html=True)
        kb2.markdown(metric_card("Exactitud", fmt_pct(best["Exactitud"]),
                                 "en datos de test", score_color(best["Exactitud"])),
                     unsafe_allow_html=True)
        kb3.markdown(metric_card("F1-Score", f"{best['F1-Score']:.3f}",
                                 "ponderado", score_color(best["F1-Score"])),
                     unsafe_allow_html=True)
        kb4.markdown(metric_card("Muestras test", len(y_test),
                                 f"de {len(y_enc)} totales", "yellow"),
                     unsafe_allow_html=True)

        if len(results) > 1:
            st.markdown("---")
            st.markdown("**Comparación de modelos**")
            comp_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("cm", "preds")}
                                     for r in results]).sort_values("Exactitud", ascending=False)
            cg1, cg2 = st.columns(2)
            with cg1:
                st.dataframe(
                    comp_df.style
                        .format({"Exactitud": "{:.2%}", "F1-Score": "{:.3f}"})
                        .background_gradient(subset=["Exactitud", "F1-Score"], cmap="Greens"),
                    use_container_width=True, hide_index=True)
            with cg2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(comp_df["Modelo"], comp_df["Exactitud"],
                        color=sns.color_palette("Set2", len(comp_df)),
                        edgecolor="white", linewidth=0.5)
                for i, val in enumerate(comp_df["Exactitud"]):
                    ax.text(val + 0.005, i, f"{val:.1%}", va="center", fontsize=9, color="white")
                ax.set_xlim(0, 1.12); ax.set_xlabel("Exactitud")
                ax.set_title("Exactitud por modelo"); ax.invert_yaxis()
                plt.tight_layout(); st.pyplot(fig); plt.clf()

        st.markdown("---")
        st.markdown(f"**Matriz de Confusión — {best['Modelo']}**")
        class_names = le.classes_.astype(str)
        cm = best["cm"]
        fig, ax = plt.subplots(figsize=(max(5, len(class_names)*1.5+2),
                                         max(4, len(class_names)*1.5+1.5)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white", ax=ax)
        ax.set_xlabel("Predicho", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        ax.set_title(f"Matriz de Confusión — {best['Modelo']}", fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.clf()

        st.markdown("---")
        st.markdown(f"**Reporte de clasificación — {best['Modelo']}**")
        report = classification_report(y_test, best["preds"],
                                        target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
        st.dataframe(
            report_df.style.format(precision=3)
                .background_gradient(subset=["precision", "recall", "f1-score"], cmap="Greens"),
            use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 · REGRESIÓN
# ════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown('<div class="section-title">Configuración</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        target_reg = st.selectbox("Variable objetivo (numérica)",
                                   [None] + df.select_dtypes(include="number").columns.tolist())
    with rc2:
        algo_reg = st.selectbox("Algoritmo", [
            "Regresión Lineal Múltiple", "Lasso", "Ridge",
            "SVM", "Árbol de Decisión", "Random Forest",
            "Gradient Boosting", "Comparar todos"
        ])

    if target_reg is None:
        st.info("Selecciona la variable objetivo para continuar.")
        st.stop()

    if st.button("▶️ Entrenar modelo(s)", type="primary", key="btn_reg"):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        df_reg = df.select_dtypes(include="number").dropna()
        X_raw = df_reg.drop(columns=[target_reg])
        y_raw = df_reg[target_reg].values
        X_sc  = StandardScaler().fit_transform(X_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X_sc, y_raw, test_size=test_size/100, random_state=random_seed)

        from sklearn.linear_model import LinearRegression, Lasso, Ridge
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.pipeline import make_pipeline

        reg_map = {
            "Regresión Lineal Múltiple": LinearRegression(),
            "Lasso":                     Lasso(alpha=0.1, max_iter=5000),
            "Ridge":                     Ridge(alpha=1.0),
            "SVM":                       make_pipeline(StandardScaler(),
                                                       SVR(kernel="rbf", C=100, epsilon=0.1)),
            "Árbol de Decisión":         DecisionTreeRegressor(max_depth=4, random_state=random_seed),
            "Random Forest":             RandomForestRegressor(n_estimators=100, max_depth=5,
                                                                random_state=random_seed),
            "Gradient Boosting":         GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                                     random_state=random_seed),
        }
        to_run = reg_map if algo_reg == "Comparar todos" else {algo_reg: reg_map[algo_reg]}

        results_reg = []
        with st.spinner("Entrenando modelos..."):
            for name, mdl in to_run.items():
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)
                results_reg.append({
                    "Modelo": name,
                    "R²":     r2_score(y_test, preds),
                    "RMSE":   math.sqrt(mean_squared_error(y_test, preds)),
                    "MAE":    mean_absolute_error(y_test, preds),
                    "preds":  preds,
                })

        best_r = max(results_reg, key=lambda r: r["R²"])
        st.markdown("---")
        st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)

        rb1, rb2, rb3, rb4 = st.columns(4)
        rb1.markdown(metric_card("Mejor Modelo", best_r["Modelo"], ""), unsafe_allow_html=True)
        rb2.markdown(metric_card("R²", f"{best_r['R²']:.4f}",
                                 "Varianza explicada", score_color(best_r["R²"])),
                     unsafe_allow_html=True)
        rb3.markdown(metric_card("RMSE", f"{best_r['RMSE']:.4f}", "Error cuadrático medio", "red"),
                     unsafe_allow_html=True)
        rb4.markdown(metric_card("MAE", f"{best_r['MAE']:.4f}", "Error absoluto medio", "yellow"),
                     unsafe_allow_html=True)

        comp_reg = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"}
                                   for r in results_reg]).sort_values("R²", ascending=False)

        rg1, rg2 = st.columns(2)
        with rg1:
            st.markdown("**Tabla comparativa**")
            st.dataframe(
                comp_reg.style
                    .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"})
                    .background_gradient(subset=["R²"], cmap="Greens")
                    .background_gradient(subset=["RMSE", "MAE"], cmap="Reds_r"),
                use_container_width=True, hide_index=True)
        with rg2:
            st.markdown("**R² por modelo**")
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#00c49a" if r["Modelo"] == best_r["Modelo"] else "#4f8bf9"
                      for r in results_reg]
            ax.barh(comp_reg["Modelo"], comp_reg["R²"],
                    color=colors[::-1], edgecolor="white", linewidth=0.5)
            ax.set_xlabel("R²"); ax.set_title("R² por modelo"); ax.invert_yaxis()
            ax.set_xlim(0, 1.1)
            for i, (val, _) in enumerate(zip(comp_reg["R²"], comp_reg["Modelo"])):
                ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9, color="white")
            plt.tight_layout(); st.pyplot(fig); plt.clf()

        st.markdown("---")
        st.markdown(f"**Predicho vs Real — {best_r['Modelo']}**")
        preds_b = best_r["preds"]
        idx = np.random.choice(len(y_test), min(len(y_test), 200), replace=False)

        fg1, fg2 = st.columns(2)
        with fg1:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test[idx], preds_b[idx], alpha=0.6, color="#4f8bf9",
                       edgecolors="white", linewidths=0.3, s=40)
            mn = min(y_test[idx].min(), preds_b[idx].min())
            mx = max(y_test[idx].max(), preds_b[idx].max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Línea perfecta")
            ax.set_xlabel("Valores reales"); ax.set_ylabel("Valores predichos")
            ax.set_title("Real vs. Predicho"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.clf()
        with fg2:
            residuals = y_test[idx] - preds_b[idx]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(residuals, bins=30, color="#4f8bf9", edgecolor="white", alpha=0.8)
            ax.axvline(0, color="#ff4b6e", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Residuo (Real − Predicho)")
            ax.set_ylabel("Frecuencia"); ax.set_title("Distribución de residuos")
            plt.tight_layout(); st.pyplot(fig); plt.clf()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ML Framework · Ciclo 1 — 2026 · Inteligencia Artificial")
