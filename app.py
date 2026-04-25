"""
app.py — Streamlit application for Emergency Access Analysis in Peru.

Run with:
    python -m streamlit run app.py

This app loads already-created files only.
It does NOT rerun data_loader, cleaning, geospatial, or metrics.
"""

from pathlib import Path
import os
import warnings

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.colors as mcolors
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
#  PAGE CONFIG
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Acceso a Emergencias — Perú",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------------ #
#  PATHS
# ------------------------------------------------------------------ #
ROOT = Path(".")
TABLES_DIR = ROOT / "output" / "tables"
FIGURES_DIR = ROOT / "output" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"


# ------------------------------------------------------------------ #
#  CONSTANTS
# ------------------------------------------------------------------ #
EAI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "eai",
    ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"],
    N=256
)


# ------------------------------------------------------------------ #
#  HELPERS
# ------------------------------------------------------------------ #
def find_existing_file(candidates):
    """Return the first existing file from a list of candidate paths."""
    for path in candidates:
        path = Path(path)
        if path.exists():
            return path
    return None


def safe_read_image_path(filename: str):
    """Return figure path if it exists, otherwise None."""
    path = FIGURES_DIR / filename
    return str(path) if path.exists() else None


@st.cache_data
def load_existing_data():
    """
    Load already-created files only.
    This function does NOT rerun any previous pipeline.
    """
    district_analysis_path = TABLES_DIR / "district_analysis.csv"
    ipress_clean_path = PROCESSED_DIR / "ipress_clean.csv"
    emergencias_clean_path = PROCESSED_DIR / "emergencias_clean.csv"

    district_geo_path = find_existing_file([
        PROCESSED_DIR / "distritos_clean.geojson",
        PROCESSED_DIR / "districts_clean.geojson",
        PROCESSED_DIR / "distritos_clean.shp",
        PROCESSED_DIR / "districts_clean.shp",
        PROCESSED_DIR / "distritos_clean.gpkg",
        PROCESSED_DIR / "districts_clean.gpkg",
    ])

    missing = []
    if not district_analysis_path.exists():
        missing.append(str(district_analysis_path))
    if not ipress_clean_path.exists():
        missing.append(str(ipress_clean_path))
    if not emergencias_clean_path.exists():
        missing.append(str(emergencias_clean_path))
    if district_geo_path is None:
        missing.append("data/processed/distritos_clean.(geojson/shp/gpkg)")

    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n- " + "\n- ".join(missing)
        )

    district_table = pd.read_csv(district_analysis_path, dtype={"ubigeo": str})
    district_table["ubigeo"] = district_table["ubigeo"].astype(str).str.zfill(6)

    distritos_clean = gpd.read_file(district_geo_path)

    if "ubigeo" not in distritos_clean.columns:
        if "IDDIST" in distritos_clean.columns:
            distritos_clean["ubigeo"] = distritos_clean["IDDIST"].astype(str).str.zfill(6)
        elif "iddist" in distritos_clean.columns:
            distritos_clean["ubigeo"] = distritos_clean["iddist"].astype(str).str.zfill(6)
        else:
            raise KeyError(
                "No recognizable UBIGEO column was found in the district geometry file."
            )
    else:
        distritos_clean["ubigeo"] = distritos_clean["ubigeo"].astype(str).str.zfill(6)

    ipress_clean = pd.read_csv(ipress_clean_path)
    emergencias_clean = pd.read_csv(emergencias_clean_path)

    # Standardize text columns that may be used later
    for df in [district_table, ipress_clean, emergencias_clean]:
        if "ubigeo" in df.columns:
            df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    return distritos_clean, district_table, ipress_clean, emergencias_clean


def create_folium_choropleth(_gdf, column, title):
    """
    Create an interactive Folium choropleth.
    No Streamlit cache here: Folium objects are not pickle-friendly.
    """
    m = folium.Map(
        location=[-9.19, -75.02],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    gdf_simple = _gdf.copy()
    gdf_simple = gdf_simple[gdf_simple[column].notna()].copy()
    gdf_simple["geometry"] = gdf_simple.geometry.simplify(0.01)

    folium.Choropleth(
        geo_data=gdf_simple.to_json(),
        data=gdf_simple,
        columns=["ubigeo", column],
        key_on="feature.properties.ubigeo",
        fill_color="RdYlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color="lightgrey",
    ).add_to(m)

    folium.GeoJson(
        gdf_simple.to_json(),
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "transparent",
            "weight": 0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["distrito", "provincia", "departamento", column],
            aliases=["Distrito:", "Provincia:", "Departamento:", f"{title}:"],
            localize=True,
        ),
    ).add_to(m)

    return m


def create_ipress_map(_ipress_df):
    """
    Create an interactive map of IPRESS points.
    No Streamlit cache here: Folium objects are not pickle-friendly.
    """
    m = folium.Map(
        location=[-9.19, -75.02],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    df = _ipress_df.copy()
    if "has_coords" in df.columns:
        df = df[df["has_coords"]].copy()

    if "lat" not in df.columns or "lon" not in df.columns:
        return m

    cluster = MarkerCluster().add_to(m)

    sample_df = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

    for _, row in sample_df.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.notna(lat) and pd.notna(lon):
            popup_name = row.get("nombre", "IPRESS")
            popup_cat = row.get("categoria", "N/A")
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="blue",
                fill=True,
                fill_opacity=0.8,
                popup=f"{popup_name}<br>Cat: {popup_cat}",
            ).add_to(cluster)

    return m


# ------------------------------------------------------------------ #
#  LOAD DATA
# ------------------------------------------------------------------ #
try:
    with st.spinner("Loading existing processed files..."):
        distritos_clean, district_table, ipress_clean, emergencias_clean = load_existing_data()
except Exception as e:
    st.error(f"Could not load required files:\n{e}")
    st.stop()


# ------------------------------------------------------------------ #
#  APP HEADER
# ------------------------------------------------------------------ #
st.title("🏥 Desigualdad en el acceso a la atención médica de emergencia — Perú")
st.markdown("*Análisis geoespacial a nivel de distrito utilizando datos abiertos de MINSA, SUSALUD, INEI e IGN.*")
st.divider()


# ------------------------------------------------------------------ #
#  TABS
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Datos y metodología",
    "📊 Análisis estático",
    "🗺️ Resultados geoespaciales",
    "🔍 Exploración interactiva",
])


# ================================================================== #
#  TAB 1 — DATOS Y METODOLOGÍA
# ================================================================== #
with tab1:
    st.header("Planteamiento del problema")
    st.markdown("""
    ¿Qué distritos de Perú parecen estar mejor o peor atendidos en términos
    de acceso a la atención médica de emergencia?

    Este proyecto combina cuatro conjuntos de datos públicos para construir un
    **Índice de Acceso a Emergencias (IAE)** a nivel de distrito que integra
    oferta territorial, capacidad estructural de emergencia, actividad observada
    y acceso espacial.
    """)

    st.header("Fuentes de datos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Conjunto de datos | Archivos | Fuente |
        |------------------|----------|--------|
        | Límites del distrito | 1,873 polígonos | INEI |
        | Centros poblados | 136,587 puntos | IGN/INEI |
        | Instalaciones IPRESS | 20,793 establecimientos | MINSA/SUSALUD |
        | Emergencias (Tabla C1) | 323,288 registros | SUSALUD 2025 |
        """)
    with col2:
        st.markdown("""
        **Sistema de coordenadas utilizado:** EPSG:4326 (WGS 84) para almacenamiento y visualización.  
        EPSG:32718 (UTM 18S) para cálculos de distancia y área.
        """)

    st.header("Resumen de limpieza")
    st.markdown("""
    - **UBIGEO:** estandarizado a 6 dígitos con ceros a la izquierda.
    - **Coordenadas de IPRESS:** los campos NORTE y ESTE estaban invertidos en la fuente y fueron corregidos.
    - **Coordenadas inválidas:** se eliminaron registros fuera de Perú.
    - **Duplicados:** se eliminaron duplicados en IPRESS y en la tabla de emergencias.
    - **Centros poblados:** se trabajaron en EPSG:4326 y se usaron para la lógica espacial.
    """)

    st.header("Metodología — Índice de Acceso a Emergencias (IAE)")
    st.markdown("""
    El índice combina cuatro componentes normalizados al rango [0, 1]:

    **Componente 1 — Oferta territorial (C_supply)**  
    Densidad de IPRESS y camas por 100 km².

    **Componente 2 — Capacidad de emergencia (C_emergency)**  
    Densidad de IPRESS hospitalarias (categorías I-4 y superiores) por 100 km².
    Estas categorías tienen mucha mayor probabilidad de reportar emergencias reales
    que las postas básicas (I-1/I-2), por lo que funcionan como proxy estructural.

    **Componente 3 — Acceso espacial (C_spatial)**  
    Distancia desde centros poblados al IPRESS más cercano, combinada con el
    porcentaje de centros poblados ubicados a más de ciertos umbrales.

    **Componente 4 — Actividad observada (C_activity)**  
    Señal de actividad de emergencia observada en Tabla C1, con peso reducido.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Especificación Baseline")
        st.latex(r"IAE_{base} = 0.35 \cdot C_{supply} + 0.25 \cdot C_{emergency} + 0.30 \cdot C_{spatial} + 0.10 \cdot C_{activity}")
        st.markdown("""
        - Más estructural y menos dependiente de C1
        - Usa distancia media y umbral de 25 km
        - Prioriza oferta y capacidad de emergencia
        """)

    with col2:
        st.subheader("Especificación Alternativa")
        st.latex(r"IAE_{alt} = 0.25 \cdot C_{supply} + 0.20 \cdot C_{emergency} + 0.35 \cdot C_{spatial} + 0.20 \cdot C_{activity}")
        st.markdown("""
        - Más exigente y más sensible a la dimensión espacial
        - Usa distancia mediana y umbral de 10 km
        - Da mayor peso a la actividad observada
        """)

    st.header("Limitaciones")
    st.markdown("""
    1. **Coordenadas IPRESS incompletas:** no todos los establecimientos tienen ubicación geográfica válida.
    2. **Distancia euclidiana:** no equivale a tiempo real de viaje por carretera o río.
    3. **Sin población distrital:** no se pueden construir tasas per cápita.
    4. **Datos de emergencia parciales:** la Tabla C1 no cubre todos los distritos por igual.
    5. **Desajuste temporal:** catálogo estático de IPRESS frente a actividad de emergencia de 2025.
    """)


# ================================================================== #
#  TAB 2 — ANÁLISIS ESTÁTICO
# ================================================================== #
with tab2:
    st.header("Visualizaciones estáticas")

    st.subheader("1. Distribución del IAE")
    fig_path = safe_read_image_path("eai_histogram.png")
    if fig_path:
        st.image(fig_path, use_container_width=True)

    st.divider()

    st.subheader("2. Distritos con mayor y menor acceso")
    fig_path = safe_read_image_path("top_bottom_districts.png")
    if fig_path:
        st.image(fig_path, use_container_width=True)

    st.divider()

    st.subheader("3. Distribución del IAE por departamento")
    fig_path = safe_read_image_path("eai_by_department.png")
    if fig_path:
        st.image(fig_path, use_container_width=True)

    st.divider()

    st.subheader("4. Densidad de instalaciones vs acceso espacial")
    fig_path = safe_read_image_path("density_vs_distance.png")
    if fig_path:
        st.image(fig_path, use_container_width=True)

    st.divider()

    st.subheader("5. Sensibilidad metodológica")
    col1, col2 = st.columns(2)
    with col1:
        fig_path = safe_read_image_path("sensitivity_scatter.png")
        if fig_path:
            st.image(fig_path, use_container_width=True)
    with col2:
        fig_path = safe_read_image_path("classification_comparison.png")
        if fig_path:
            st.image(fig_path, use_container_width=True)

    if {"eai_baseline", "eai_alternative"}.issubset(district_table.columns):
        corr = district_table[["eai_baseline", "eai_alternative"]].corr().iloc[0, 1]
        n_changed = (district_table["quintil_baseline"] != district_table["quintil_alternative"]).sum()

        st.markdown(f"""
        **Interpretación:** La correlación entre ambas especificaciones es **{corr:.3f}**,
        lo que sugiere alta coherencia general. Sin embargo, **{n_changed} distritos**
        (**{100*n_changed/len(district_table):.1f}%**) cambian de quintil, lo que refleja
        sensibilidad metodológica en una parte no menor del país.
        """)


# ================================================================== #
#  TAB 3 — RESULTADOS GEOESPACIALES
# ================================================================== #
with tab3:
    st.header("Mapas estáticos y comparación distrital")

    st.subheader("Mapa coroplético — IAE Baseline")
    fig_path = safe_read_image_path("choropleth_eai_baseline.png")
    if fig_path:
        st.image(fig_path, use_container_width=True)

    st.divider()

    st.subheader("Mapa coroplético — IAE Alternativo")
    fig_path = safe_read_image_path("choropleth_eai_alternative.png")
    if fig_path:
        st.image(fig_path, use_container_width=True)

    st.divider()

    st.subheader("Tabla de comparación distrital")
    depts = sorted(district_table["departamento"].dropna().unique().tolist())
    selected_dept = st.selectbox("Filtrar por departamento:", ["Todos"] + depts)

    display_df = district_table.copy()
    if selected_dept != "Todos":
        display_df = display_df[display_df["departamento"] == selected_dept]

    sort_col = st.selectbox(
        "Ordenar por:",
        ["eai_baseline", "eai_alternative", "mean_dist_km", "n_ipress", "total_atendidos"]
    )
    ascending = st.checkbox("Orden ascendente", value=True)
    display_df = display_df.sort_values(sort_col, ascending=ascending)

    show_cols = [
        "distrito", "provincia", "departamento",
        "n_ipress", "n_ipress_hospitalaria", "n_camas",
        "total_atendidos", "mean_dist_km",
        "eai_baseline", "quintil_baseline",
        "eai_alternative", "quintil_alternative"
    ]
    show_cols = [c for c in show_cols if c in display_df.columns]

    st.dataframe(
        display_df[show_cols].head(50),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Mostrando {min(50, len(display_df))} de {len(display_df)} distritos")


# ================================================================== #
#  TAB 4 — EXPLORACIÓN INTERACTIVA
# ================================================================== #
with tab4:
    st.header("Exploración interactiva con Folium")

    map_type = st.radio(
        "Tipo de mapa:",
        ["IAE Baseline", "IAE Alternativo", "Instalaciones IPRESS"],
        horizontal=True
    )

    gdf_map = distritos_clean.merge(
        district_table[
            [
                "ubigeo", "eai_baseline", "eai_alternative",
                "distrito", "provincia", "departamento",
                "n_ipress", "mean_dist_km"
            ]
        ],
        on="ubigeo",
        how="left",
        suffixes=("_geo", "")
    )

    for col in ["distrito_geo", "provincia_geo", "departamento_geo"]:
        if col in gdf_map.columns:
            gdf_map = gdf_map.drop(columns=col)

    if map_type == "IAE Baseline":
        m = create_folium_choropleth(gdf_map, "eai_baseline", "IAE Baseline")
    elif map_type == "IAE Alternativo":
        m = create_folium_choropleth(gdf_map, "eai_alternative", "IAE Alternativo")
    else:
        m = create_ipress_map(ipress_clean)

    st_folium(m, width=None, height=600, use_container_width=True)

    st.divider()

    st.subheader("Comparador de distritos")

    district_labels_df = district_table[["distrito", "provincia", "departamento", "ubigeo"]].copy()
    district_labels_df["label"] = district_labels_df.apply(
        lambda r: f"{r['distrito']} ({r['provincia']}, {r['departamento']})",
        axis=1
    )
    district_labels_df = district_labels_df.sort_values("label")

    district_labels = district_labels_df["label"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        dist1_label = st.selectbox("Distrito 1:", district_labels, index=0)
    with col2:
        dist2_label = st.selectbox("Distrito 2:", district_labels, index=min(len(district_labels) - 1, 1))

    dist1_row = district_labels_df[district_labels_df["label"] == dist1_label].iloc[0]
    dist2_row = district_labels_df[district_labels_df["label"] == dist2_label].iloc[0]

    d1 = district_table[district_table["ubigeo"] == dist1_row["ubigeo"]].iloc[0]
    d2 = district_table[district_table["ubigeo"] == dist2_row["ubigeo"]].iloc[0]

    compare_metrics = [
        "n_ipress",
        "n_ipress_hospitalaria",
        "n_camas",
        "total_atendidos",
        "mean_dist_km",
        "pct_beyond_25km",
        "n_ccpp",
        "eai_baseline",
        "eai_alternative",
        "quintil_baseline",
        "quintil_alternative",
    ]
    compare_labels = [
        "Nº IPRESS",
        "IPRESS hospitalarias",
        "Nº camas",
        "Atendidos emergencia",
        "Distancia media (km)",
        "% CCPP > 25 km",
        "Centros poblados",
        "IAE Baseline",
        "IAE Alternativo",
        "Quintil Baseline",
        "Quintil Alternativo",
    ]

    comparison_data = []
    for metric, label in zip(compare_metrics, compare_labels):
        if metric in district_table.columns:
            v1 = d1.get(metric, "N/A")
            v2 = d2.get(metric, "N/A")
            if isinstance(v1, float):
                v1 = f"{v1:.3f}"
                v2 = f"{v2:.3f}"
            comparison_data.append({
                "Indicador": label,
                dist1_row["distrito"]: v1,
                dist2_row["distrito"]: v2
            })

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Distritos con mayor sensibilidad metodológica")

    if {"quintil_baseline", "quintil_alternative", "abs_rank_change"}.issubset(district_table.columns):
        changed = district_table[
            district_table["quintil_baseline"] != district_table["quintil_alternative"]
        ].copy()
        changed = changed.sort_values("abs_rank_change", ascending=False)

        show = changed[
            [
                "distrito", "provincia", "departamento",
                "eai_baseline", "quintil_baseline",
                "eai_alternative", "quintil_alternative",
                "rank_change"
            ]
        ].head(20)

        st.dataframe(show, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ #
#  FOOTER
# ------------------------------------------------------------------ #
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        Proyecto: Desigualdad en el acceso a atención médica de emergencia en Perú<br>
        Datos: MINSA, SUSALUD, INEI, IGN — 2025<br>
        Herramientas: Python, GeoPandas, Folium, Streamlit
    </div>
    """,
    unsafe_allow_html=True
)