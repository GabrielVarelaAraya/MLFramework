import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib

from util_pca import ACPmain


def capture_print(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


st.title('Framework IA — Interfaz básica')

uploaded_file = st.file_uploader('Carga un CSV', type=['csv'])
sep = st.selectbox('Delimitador', [',', ';'], index=0)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success('Archivo cargado correctamente')
    except Exception as e:
        st.error(f'Error al leer el CSV: {e}')
else:
    st.info('Sube un archivo CSV para empezar o ejecuta desde tu script.')

if df is not None:
    st.subheader('Vista previa')
    st.dataframe(df.head())

    st.sidebar.header('Acciones EDA')
    if st.sidebar.button('Mostrar estadísticas'):
        st.write(df.describe(include='all'))

    if st.sidebar.button('Valores nulos'): 
        st.write(df.isnull().sum())

    col_uni = st.sidebar.text_input('Columna para valores únicos (opcional)')
    if col_uni and st.sidebar.button('Mostrar valores únicos'):
        if col_uni in df.columns:
            st.write(df[col_uni].value_counts())
        else:
            st.error('Columna no encontrada')

    if st.sidebar.button('Boxplot (variables numéricas)'):
        try:
            from eda import analisisEDA
            eda = analisisEDA(None, None)
            eda.df = df.select_dtypes(include=['number'])
            fig = plt.figure(figsize=(8, 4))
            eda.graficoBoxplot()
            st.pyplot(fig)
            plt.clf()
        except Exception as e:
            st.error(f'Error generando boxplot: {e}')

    st.sidebar.header('PCA / ACP')
    n_comp = st.sidebar.number_input('N componentes', min_value=2, max_value=10, value=2)
    if st.sidebar.button('Ejecutar ACP y mostrar gráficos'):
        try:
            df_num = df.select_dtypes(include=['number']).dropna()
            acp = ACPmain(df_num, n_comp)
            # Plano principal
            fig = plt.figure(figsize=(6, 4))
            acp.plot_plano_principal()
            st.pyplot(fig)
            plt.clf()
            # Círculo de correlación
            fig = plt.figure(figsize=(6, 6))
            acp.plot_circulo()
            st.pyplot(fig)
            plt.clf()
            # Sobreposición
            fig = plt.figure(figsize=(6, 4))
            acp.plot_sobreposicion()
            st.pyplot(fig)
            plt.clf()
        except Exception as e:
            st.error(f'Error ejecutando ACP: {e}')

    st.sidebar.header('Clustering simple')
    n_clusters = st.sidebar.slider('Número de clusters (KMeans)', 2, 10, 3)
    if st.sidebar.button('Ejecutar KMeans (2D)'):
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            df_num = df.select_dtypes(include=['number']).dropna()
            pca = PCA(n_components=2)
            comps = pca.fit_transform(df_num)
            kmed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmed.fit_predict(df_num)
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(comps[:, 0], comps[:, 1], c=labels, cmap='tab10', alpha=0.8)
            plt.title(f'KMeans (k={n_clusters})')
            st.pyplot(fig)
            plt.clf()
        except Exception as e:
            st.error(f'Error en KMeans: {e}')

    st.sidebar.header('Clasificación (básica)')
    target = st.sidebar.selectbox('Selecciona la columna target', options=[None] + list(df.columns))
    knn_neighbors = st.sidebar.slider('Vecinos KNN', 1, 15, 3)
    if st.sidebar.button('Entrenar KNN (usa clase Clasificacion)'):
        if target is None or target not in df.columns:
            st.error('Selecciona una columna target válida')
        else:
            try:
                # Importar localmente para evitar dependencias pesadas al arrancar
                from clasificacion import Clasificacion
                df_clf = df.copy()
                clf = Clasificacion(df_clf)
                out = capture_print(clf.KNN, knn_neighbors)
                st.text(out)
            except Exception as e:
                st.error(f'Error en Clasificación: {e}')

    st.sidebar.header('Exportar')
    if st.sidebar.button('Descargar CSV limpiado'):
        to_download = df.to_csv(index=False).encode('utf-8')
        st.download_button('Descargar', to_download, file_name='data_clean.csv')

st.markdown('---')
st.caption('Interfaz sencilla creada con Streamlit — añade más funciones según lo necesites')
