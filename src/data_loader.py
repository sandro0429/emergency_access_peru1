"""
data_loader.py — Functions to load the four required datasets.

Datasets:
    1. DISTRITOS.shp — District boundaries (polygons)
    2. CCPP_IGN100K.shp — Populated centers (points)
    3. IPRESS.csv — Health facilities
    4. ConsultaC1_2025_v20.csv — Emergency production by IPRESS
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

from src.utils import RAW_DIR, print_summary


def load_distritos(path: str | Path = None) -> gpd.GeoDataFrame:
    """
    Load the district boundaries shapefile.

    Source: INEI / GeoGPS Perú
    CRS expected: EPSG:4326 (WGS 84)
    Key columns: IDDIST (ubigeo 6-digit), DEPARTAMEN, PROVINCIA, DISTRITO
    """
    if path is None:
        path = RAW_DIR / "DISTRITOS.shp"
    else:
        path = Path(path)

    gdf = gpd.read_file(path)
    print(f"[load_distritos] Loaded {len(gdf)} districts, CRS={gdf.crs}")
    return gdf


def load_centros_poblados(path: str | Path = None) -> gpd.GeoDataFrame:
    """
    Load the populated centers shapefile.

    Source: IGN / Datos Abiertos
    CRS expected: EPSG:4326 (WGS 84)
    Key columns: NOM_POBLAD, CÓDIGO (10-digit, first 6 = ubigeo),
                 DEP, PROV, DIST, CATEGORIA, X, Y
    """
    if path is None:
        path = RAW_DIR / "CCPP_IGN100K.shp"
    else:
        path = Path(path)

    gdf = gpd.read_file(path)
    print(f"[load_centros_poblados] Loaded {len(gdf)} centers, CRS={gdf.crs}")
    return gdf


def load_ipress(path: str | Path = None) -> pd.DataFrame:
    """
    Load the IPRESS health facilities CSV.

    Source: MINSA / SUSALUD via Datos Abiertos
    Encoding: latin-1
    """
    if path is None:
        path = RAW_DIR / "IPRESS.csv"
    else:
        path = Path(path)

    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    print(f"[load_ipress] Loaded {len(df)} facilities")
    return df


def load_emergencias(path: str | Path = None) -> pd.DataFrame:
    """
    Load the emergency production data (Tabla C1 - SUSALUD).

    Source: SUSALUD — Producción Asistencial en Emergencia por IPRESS
    Encoding: latin-1, separator: semicolon (;)
    """
    if path is None:
        path = RAW_DIR / "ConsultaC1_2025_v20.csv"
    else:
        path = Path(path)

    df = pd.read_csv(path, encoding="latin1", sep=";", low_memory=False)
    print(f"[load_emergencias] Loaded {len(df)} records")
    return df


def load_all(raw_dir: str | Path = None) -> tuple:
    """
    Convenience function: load all four datasets at once.

    Returns:
        (distritos_gdf, ccpp_gdf, ipress_df, emergencias_df)
    """
    if raw_dir:
        raw_dir = Path(raw_dir)
        return (
            load_distritos(raw_dir / "DISTRITOS.shp"),
            load_centros_poblados(raw_dir / "CCPP_IGN100K.shp"),
            load_ipress(raw_dir / "IPRESS.csv"),
            load_emergencias(raw_dir / "ConsultaC1_2025_v20.csv"),
        )

    return (
        load_distritos(),
        load_centros_poblados(),
        load_ipress(),
        load_emergencias(),
    )


def run_loader_check() -> None:
    """Load all datasets and print short summaries."""
    print("=" * 60)
    print("RUNNING DATA LOADER CHECK")
    print("=" * 60)
    print(f"RAW_DIR = {RAW_DIR}")

    distritos, ccpp, ipress, emergencias = load_all()

    print_summary(distritos, "DISTRITOS")
    print_summary(ccpp, "CCPP")
    print_summary(ipress, "IPRESS")
    print_summary(emergencias, "EMERGENCIAS")


if __name__ == "__main__":
    run_loader_check()