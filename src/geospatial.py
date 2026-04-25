"""
geospatial.py — Integración geoespacial, joins espaciales y lógica de distancias.

Estrategia de CRS:
    - Todos los datos limpios se trabajan en EPSG:4326 para compatibilidad con mapas.
    - Para cálculos de distancia y área se proyecta a EPSG:32718 (UTM 18S),
      que entrega distancias en metros y funciona razonablemente bien para Perú.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree

from src.utils import ensure_dir
from src.cleaning import (
    run_cleaning_pipeline,
    aggregate_emergencias_by_district,
)

PROCESSED_DIR = Path("data") / "processed"
OUTPUT_TABLES_DIR = Path("output") / "tables"
ensure_dir(PROCESSED_DIR)
ensure_dir(OUTPUT_TABLES_DIR)


# ------------------------------------------------------------------ #
#  HELPERS DE GUARDADO (SOBRESCRIBIR)
# ------------------------------------------------------------------ #
def save_geojson_overwrite(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Guarda un GeoDataFrame en GeoJSON, sobrescribiendo si ya existe."""
    path = Path(path)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GeoJSON")
    print(f"[save_geojson_overwrite] Guardado: {path}")


def save_csv_overwrite(df: pd.DataFrame, path: Path) -> None:
    """Guarda un DataFrame en CSV, sobrescribiendo si ya existe."""
    path = Path(path)
    df.to_csv(path, index=False)
    print(f"[save_csv_overwrite] Guardado: {path}")


# ------------------------------------------------------------------ #
#  CREAR GEODATAFRAME DESDE IPRESS TABULAR
# ------------------------------------------------------------------ #
def ipress_to_geodataframe(ipress_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convierte el DataFrame limpio de IPRESS a GeoDataFrame.

    Solo usa establecimientos con coordenadas válidas.
    """
    df_with_coords = ipress_df[ipress_df["has_coords"]].copy()
    geometry = [Point(xy) for xy in zip(df_with_coords["lon"], df_with_coords["lat"])]
    gdf = gpd.GeoDataFrame(df_with_coords, geometry=geometry, crs="EPSG:4326")

    print(
        f"[ipress_to_geodataframe] {len(gdf)} IPRESS con coordenadas válidas "
        f"(de {len(ipress_df)} registros totales)"
    )
    return gdf


# ------------------------------------------------------------------ #
#  JOIN ESPACIAL: ASIGNAR PUNTOS A DISTRITOS
# ------------------------------------------------------------------ #
def assign_to_districts(
    points_gdf: gpd.GeoDataFrame,
    districts_gdf: gpd.GeoDataFrame,
    label: str = "puntos"
) -> gpd.GeoDataFrame:
    """
    Asigna puntos a distritos mediante join espacial punto-en-polígono.

    Agrega la columna 'ubigeo_spatial' proveniente del polígono distrital.
    """
    if points_gdf.crs != districts_gdf.crs:
        points_gdf = points_gdf.to_crs(districts_gdf.crs)

    result = gpd.sjoin(
        points_gdf,
        districts_gdf[["ubigeo", "geometry"]].rename(columns={"ubigeo": "ubigeo_spatial"}),
        how="left",
        predicate="within"
    )

    n_matched = result["ubigeo_spatial"].notna().sum()
    n_total = len(result)
    print(f"[assign_to_districts] {label}: {n_matched}/{n_total} puntos asignados a un distrito")

    if "index_right" in result.columns:
        result = result.drop(columns="index_right")

    return result


def assign_ccpp_to_districts(
    ccpp_gdf: gpd.GeoDataFrame,
    districts_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Asigna centros poblados a distritos por join espacial.

    Si falla el join espacial, usa 'ubigeo_codigo' como respaldo.
    """
    result = assign_to_districts(ccpp_gdf, districts_gdf, label="CCPP")

    if "ubigeo_codigo" in result.columns:
        mask_no_spatial = result["ubigeo_spatial"].isna() & result["ubigeo_codigo"].notna()
        n_fallback = mask_no_spatial.sum()

        if n_fallback > 0:
            result.loc[mask_no_spatial, "ubigeo_spatial"] = result.loc[mask_no_spatial, "ubigeo_codigo"]
            print(f"[assign_ccpp_to_districts] Se usó ubigeo_codigo como respaldo para {n_fallback} CCPP")

    result["ubigeo_final"] = result["ubigeo_spatial"].fillna(result.get("ubigeo_codigo"))
    return result


def assign_ipress_to_districts(
    ipress_gdf: gpd.GeoDataFrame,
    districts_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Asigna IPRESS a distritos por join espacial.

    Mantiene el UBIGEO original del CSV como referencia principal.
    """
    result = assign_to_districts(ipress_gdf, districts_gdf, label="IPRESS")
    result["ubigeo_validated"] = result["ubigeo"]
    return result


# ------------------------------------------------------------------ #
#  DISTANCIA AL IPRESS MÁS CERCANO
# ------------------------------------------------------------------ #
def compute_nearest_ipress_distance(
    ccpp_gdf: gpd.GeoDataFrame,
    ipress_gdf: gpd.GeoDataFrame
) -> pd.Series:
    """
    Calcula la distancia en km desde cada centro poblado
    al IPRESS más cercano usando KD-tree.
    """
    ccpp_proj = ccpp_gdf.to_crs("EPSG:32718")
    ipress_proj = ipress_gdf.to_crs("EPSG:32718")

    ipress_coords = np.array(list(zip(ipress_proj.geometry.x, ipress_proj.geometry.y)))
    tree = cKDTree(ipress_coords)

    ccpp_coords = np.array(list(zip(ccpp_proj.geometry.x, ccpp_proj.geometry.y)))
    distances, _ = tree.query(ccpp_coords, k=1)

    distances_km = distances / 1000.0

    print(
        "[compute_nearest_ipress_distance] "
        f"media={distances_km.mean():.2f} km, "
        f"mediana={np.median(distances_km):.2f} km, "
        f"máx={distances_km.max():.2f} km"
    )

    return pd.Series(distances_km, index=ccpp_gdf.index, name="dist_nearest_ipress_km")


def compute_ccpp_distances_by_district(
    ccpp_with_dist: gpd.GeoDataFrame,
    ubigeo_col: str = "ubigeo_final"
) -> pd.DataFrame:
    """
    Agrega métricas de distancia CCPP-IPRESS a nivel distrital.
    """
    df = ccpp_with_dist.copy()
    df = df[df[ubigeo_col].notna() & df["dist_nearest_ipress_km"].notna()]

    agg = df.groupby(ubigeo_col).agg(
        mean_dist_km=("dist_nearest_ipress_km", "mean"),
        median_dist_km=("dist_nearest_ipress_km", "median"),
        max_dist_km=("dist_nearest_ipress_km", "max"),
        n_ccpp=("dist_nearest_ipress_km", "count"),
    ).reset_index()

    agg = agg.rename(columns={ubigeo_col: "ubigeo"})

    beyond_10 = df.groupby(ubigeo_col).apply(
        lambda g: (g["dist_nearest_ipress_km"] > 10).mean(),
        include_groups=False
    ).reset_index(name="pct_beyond_10km").rename(columns={ubigeo_col: "ubigeo"})

    beyond_25 = df.groupby(ubigeo_col).apply(
        lambda g: (g["dist_nearest_ipress_km"] > 25).mean(),
        include_groups=False
    ).reset_index(name="pct_beyond_25km").rename(columns={ubigeo_col: "ubigeo"})

    agg = agg.merge(beyond_10, on="ubigeo", how="left")
    agg = agg.merge(beyond_25, on="ubigeo", how="left")

    print(f"[compute_ccpp_distances_by_district] Métricas calculadas para {len(agg)} distritos")
    return agg


# ------------------------------------------------------------------ #
#  CONTEO DE IPRESS POR DISTRITO
# ------------------------------------------------------------------ #
EMERGENCY_CAPABLE_CATEGORIES = [
    "I-4", "II-1", "II-2", "II-E", "III-1", "III-2", "III-E"
]


def count_ipress_by_district(ipress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cuenta IPRESS por distrito usando UBIGEO tabular.

    Incluye:
        - n_ipress
        - n_camas
        - n_con_internamiento
        - n_ipress_hospitalaria
    """
    df = ipress_df.copy()
    df["es_hospitalaria"] = df["categoria"].isin(EMERGENCY_CAPABLE_CATEGORIES)

    agg = df.groupby("ubigeo").agg(
        n_ipress=("codigo_unico", "nunique"),
        n_camas=("camas", "sum"),
        n_con_internamiento=("tipo", lambda x: x.astype(str).str.contains("CON INTERNAMIENTO", na=False).sum()),
        n_ipress_hospitalaria=("es_hospitalaria", "sum"),
    ).reset_index()

    n_with_hosp = (agg["n_ipress_hospitalaria"] > 0).sum()
    print(
        f"[count_ipress_by_district] {len(agg)} distritos con instalaciones; "
        f"{n_with_hosp} tienen al menos una IPRESS hospitalaria/especializada"
    )
    return agg


# ------------------------------------------------------------------ #
#  PIPELINE COMPLETO GEOESPACIAL
# ------------------------------------------------------------------ #
def run_geospatial_pipeline():
    """
    Ejecuta el pipeline completo geoespacial y guarda outputs.
    """
    print("\n" + "=" * 60)
    print("EJECUTANDO PIPELINE GEOESPACIAL")
    print("=" * 60)

    distritos_clean, ccpp_clean, ipress_clean, emergencias_clean, _ = run_cleaning_pipeline()

    ipress_gdf = ipress_to_geodataframe(ipress_clean)
    ccpp_districts = assign_ccpp_to_districts(ccpp_clean, distritos_clean)
    ipress_districts = assign_ipress_to_districts(ipress_gdf, distritos_clean)

    ccpp_districts["dist_nearest_ipress_km"] = compute_nearest_ipress_distance(ccpp_districts, ipress_gdf)

    district_distances = compute_ccpp_distances_by_district(ccpp_districts)
    district_ipress_count = count_ipress_by_district(ipress_clean)
    district_emergencias = aggregate_emergencias_by_district(emergencias_clean)

    save_geojson_overwrite(ccpp_districts, PROCESSED_DIR / "ccpp_districts.geojson")
    save_geojson_overwrite(ipress_districts, PROCESSED_DIR / "ipress_districts.geojson")
    save_csv_overwrite(district_distances, OUTPUT_TABLES_DIR / "district_distances.csv")
    save_csv_overwrite(district_ipress_count, OUTPUT_TABLES_DIR / "district_ipress_count.csv")
    save_csv_overwrite(district_emergencias, OUTPUT_TABLES_DIR / "district_emergencias.csv")

    print(f"[run_geospatial_pipeline] Outputs guardados en {PROCESSED_DIR} y {OUTPUT_TABLES_DIR}")

    return {
        "ipress_gdf": ipress_gdf,
        "ccpp_districts": ccpp_districts,
        "ipress_districts": ipress_districts,
        "district_distances": district_distances,
        "district_ipress_count": district_ipress_count,
        "district_emergencias": district_emergencias,
    }


if __name__ == "__main__":
    run_geospatial_pipeline()