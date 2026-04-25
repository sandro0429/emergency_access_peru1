"""
cleaning.py — Data cleaning and preprocessing pipeline.

Cleaning decisions documented here:
    1. UBIGEO standardization: zero-pad to 6 digits across all datasets.
    2. IPRESS coordinates: swap NORTE/ESTE (mislabeled in source data).
    3. IPRESS coordinate validation: keep only points within Peru's bounding box.
    4. CCPP: set CRS to EPSG:4326 when missing.
    5. Duplicates: remove exact duplicates in each dataset.
    6. Column renaming: standardize to snake_case for consistency.
"""
print("SE ABRIO cleaning.py")

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

from src.utils import pad_ubigeo, normalize_text, ensure_dir
from src.data_loader import (
    load_distritos,
    load_centros_poblados,
    load_ipress,
    load_emergencias,
)

# Directorio de salida
PROCESSED_DIR = Path("data") / "processed"
ensure_dir(PROCESSED_DIR)

# Peru's approximate bounding box (WGS 84)
PERU_BOUNDS = {
    "lon_min": -81.5,
    "lon_max": -68.5,
    "lat_min": -18.4,
    "lat_max": 0.1,
}


# ------------------------------------------------------------------ #
#  DISTRITOS
# ------------------------------------------------------------------ #
def clean_distritos(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf = gdf.rename(columns={
        "IDDPTO": "id_dpto",
        "DEPARTAMEN": "departamento",
        "IDPROV": "id_prov",
        "PROVINCIA": "provincia",
        "IDDIST": "ubigeo",
        "DISTRITO": "distrito",
        "CAPITAL": "capital",
        "AREA": "area_tipo",
    })

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["ubigeo"] = pad_ubigeo(gdf["ubigeo"])
    gdf["departamento"] = normalize_text(gdf["departamento"])
    gdf["provincia"] = normalize_text(gdf["provincia"])
    gdf["distrito"] = normalize_text(gdf["distrito"])

    gdf["area_km2"] = gdf.to_crs("EPSG:32718").geometry.area / 1e6

    n_null = gdf.geometry.isna().sum()
    if n_null > 0:
        print(f"[clean_distritos] Dropping {n_null} null geometries")
        gdf = gdf[gdf.geometry.notna()].copy()

    n_dup = gdf.duplicated(subset="ubigeo").sum()
    if n_dup > 0:
        print(f"[clean_distritos] Dropping {n_dup} duplicate ubigeos")
        gdf = gdf.drop_duplicates(subset="ubigeo").copy()

    cols = ["ubigeo", "departamento", "provincia", "distrito",
            "capital", "area_km2", "geometry"]
    gdf = gdf[[c for c in cols if c in gdf.columns]]

    print(f"[clean_distritos] Result: {len(gdf)} districts")
    return gdf


# ------------------------------------------------------------------ #
#  CENTROS POBLADOS
# ------------------------------------------------------------------ #
def clean_centros_poblados(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Extraer ubigeo desde CÓDIGO
    gdf["ubigeo_codigo"] = gdf["CÓDIGO"].astype(str).str[:6]
    gdf.loc[gdf["CÓDIGO"].isna(), "ubigeo_codigo"] = None

    gdf = gdf.rename(columns={
        "NOM_POBLAD": "nombre",
        "CÓDIGO": "codigo",
        "CAT_POBLAD": "cat_poblado",
        "DIST": "distrito",
        "PROV": "provincia",
        "DEP": "departamento",
        "CATEGORIA": "categoria",
        "X": "lon",
        "Y": "lat",
    })

    for col in ["nombre", "departamento", "provincia", "distrito"]:
        if col in gdf.columns:
            gdf[col] = normalize_text(gdf[col])

    gdf["ubigeo_codigo"] = pad_ubigeo(gdf["ubigeo_codigo"])

    n_null = gdf.geometry.isna().sum()
    if n_null > 0:
        print(f"[clean_ccpp] Dropping {n_null} null geometries")
        gdf = gdf[gdf.geometry.notna()].copy()

    gdf["_x"] = gdf.geometry.x
    gdf["_y"] = gdf.geometry.y
    in_peru = (
        (gdf["_x"] >= PERU_BOUNDS["lon_min"]) &
        (gdf["_x"] <= PERU_BOUNDS["lon_max"]) &
        (gdf["_y"] >= PERU_BOUNDS["lat_min"]) &
        (gdf["_y"] <= PERU_BOUNDS["lat_max"])
    )
    n_out = (~in_peru).sum()
    if n_out > 0:
        print(f"[clean_ccpp] Dropping {n_out} points outside Peru")
        gdf = gdf[in_peru].copy()
    gdf = gdf.drop(columns=["_x", "_y"])

    cols = ["nombre", "codigo", "ubigeo_codigo", "cat_poblado",
            "departamento", "provincia", "distrito", "categoria",
            "lon", "lat", "geometry"]
    gdf = gdf[[c for c in cols if c in gdf.columns]]

    print(f"[clean_ccpp] Result: {len(gdf)} populated centers")
    return gdf


# ------------------------------------------------------------------ #
#  IPRESS
# ------------------------------------------------------------------ #
def clean_ipress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["UBIGEO"] = pad_ubigeo(df["UBIGEO"])
    df["Código Único"] = df["Código Único"].astype(str).str.strip().str.zfill(8)

    # NORTE = longitud, ESTE = latitud
    df["lon"] = pd.to_numeric(df["NORTE"], errors="coerce")
    df["lat"] = pd.to_numeric(df["ESTE"], errors="coerce")

    valid_coords = (
        df["lon"].notna() &
        df["lat"].notna() &
        (df["lon"] >= PERU_BOUNDS["lon_min"]) &
        (df["lon"] <= PERU_BOUNDS["lon_max"]) &
        (df["lat"] >= PERU_BOUNDS["lat_min"]) &
        (df["lat"] <= PERU_BOUNDS["lat_max"])
    )

    n_invalid = (~valid_coords & df["lon"].notna()).sum()
    n_missing = df["lon"].isna().sum()
    print(f"[clean_ipress] Coords: {valid_coords.sum()} valid, "
          f"{n_missing} missing, {n_invalid} invalid")

    df["has_coords"] = valid_coords
    df.loc[~valid_coords, ["lon", "lat"]] = np.nan

    df = df.rename(columns={
        "Código Único": "codigo_unico",
        "Nombre del establecimiento": "nombre",
        "Institución": "institucion",
        "Clasificación": "clasificacion",
        "Tipo": "tipo",
        "Departamento": "departamento",
        "Provincia": "provincia",
        "Distrito": "distrito",
        "UBIGEO": "ubigeo",
        "Categoria": "categoria",
        "CAMAS": "camas",
        "Estado": "estado",
        "Condición": "condicion",
    })

    for col in ["departamento", "provincia", "distrito", "nombre"]:
        if col in df.columns:
            df[col] = normalize_text(df[col])

    df["camas"] = pd.to_numeric(df["camas"], errors="coerce").fillna(0).astype(int)

    n_dup = df.duplicated(subset="codigo_unico").sum()
    if n_dup > 0:
        print(f"[clean_ipress] Dropping {n_dup} duplicate facilities")
        df = df.drop_duplicates(subset="codigo_unico").copy()

    cols = ["codigo_unico", "nombre", "institucion", "clasificacion",
            "tipo", "departamento", "provincia", "distrito", "ubigeo",
            "categoria", "camas", "lon", "lat", "has_coords",
            "estado", "condicion"]
    df = df[[c for c in cols if c in df.columns]]

    print(f"[clean_ipress] Result: {len(df)} facilities")
    return df


# ------------------------------------------------------------------ #
#  EMERGENCIAS
# ------------------------------------------------------------------ #
def clean_emergencias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["UBIGEO"] = pad_ubigeo(df["UBIGEO"])
    df["CO_IPRESS"] = df["CO_IPRESS"].astype(str).str.strip().str.zfill(8)

    for col in ["NRO_TOTAL_ATENCIONES", "NRO_TOTAL_ATENDIDOS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.rename(columns={
        "ANHO": "anio",
        "MES": "mes",
        "UBIGEO": "ubigeo",
        "DEPARTAMENTO": "departamento",
        "PROVINCIA": "provincia",
        "DISTRITO": "distrito",
        "SECTOR": "sector",
        "CATEGORIA": "categoria",
        "CO_IPRESS": "codigo_ipress",
        "RAZON_SOC": "razon_social",
        "SEXO": "sexo",
        "EDAD": "grupo_edad",
        "NRO_TOTAL_ATENCIONES": "total_atenciones",
        "NRO_TOTAL_ATENDIDOS": "total_atendidos",
    })

    for col in ["departamento", "provincia", "distrito"]:
        df[col] = normalize_text(df[col])

    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(f"[clean_emergencias] Dropping {n_dup} exact duplicates")
        df = df.drop_duplicates().copy()

    print(f"[clean_emergencias] Result: {len(df)} records, "
          f"{df['codigo_ipress'].nunique()} IPRESS, "
          f"{df['ubigeo'].nunique()} districts")
    return df


def aggregate_emergencias_by_district(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("ubigeo").agg(
        total_atenciones=("total_atenciones", "sum"),
        total_atendidos=("total_atendidos", "sum"),
        n_ipress_emergencia=("codigo_ipress", "nunique"),
    ).reset_index()

    print(f"[agg_emergencias] {len(agg)} districts with emergency data")
    return agg


# ------------------------------------------------------------------ #
#  RUN FULL CLEANING PIPELINE
# ------------------------------------------------------------------ #
def run_cleaning_pipeline(save_dir=None):
    if save_dir is None:
        save_dir = PROCESSED_DIR

    print("=" * 60)
    print("RUNNING CLEANING PIPELINE")
    print("=" * 60)

    distritos_raw = load_distritos()
    ccpp_raw = load_centros_poblados()
    ipress_raw = load_ipress()
    emergencias_raw = load_emergencias()

    d = clean_distritos(distritos_raw)
    c = clean_centros_poblados(ccpp_raw)
    i = clean_ipress(ipress_raw)
    e = clean_emergencias(emergencias_raw)
    e_dist = aggregate_emergencias_by_district(e)

    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    d.to_file(save_dir / "distritos_clean.geojson", driver="GeoJSON")
    c.to_file(save_dir / "ccpp_clean.geojson", driver="GeoJSON")
    i.to_csv(save_dir / "ipress_clean.csv", index=False)
    e.to_csv(save_dir / "emergencias_clean.csv", index=False)
    e_dist.to_csv(save_dir / "emergencias_distrito.csv", index=False)

    print(f"\n[pipeline] Cleaned datasets saved to {save_dir}/")
    return d, c, i, e, e_dist


if __name__ == "__main__":
    from pathlib import Path
    from src.utils import ensure_dir
    from src.data_loader import (
        load_distritos,
        load_centros_poblados,
        load_ipress,
        load_emergencias,
    )

    PROCESSED_DIR = Path("data") / "processed"
    ensure_dir(PROCESSED_DIR)

    run_cleaning_pipeline(PROCESSED_DIR)

    
if __name__ == "__main__":
    print("SE ESTA EJECUTANDO run_cleaning_pipeline()")
    run_cleaning_pipeline()    