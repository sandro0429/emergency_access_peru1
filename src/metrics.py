"""
metrics.py — District-level access metrics, composite index, and sensitivity analysis.

===============================================================================
METHODOLOGICAL FRAMEWORK
===============================================================================

OBJECTIVE
    Construct a district-level Emergency Access Index (EAI) to measure how
    well-served each of Peru's 1,873 districts is in terms of emergency
    healthcare access.

ANALYTICAL LOGIC
    The index is designed to answer four questions:

    Q1. Territorial availability:
        Which districts appear to have lower or higher availability of
        health facilities and emergency-related services?

    Q2. Settlement access:
        Which districts appear to have populated centers with weaker spatial
        access to emergency-related health services?

    Q3. District comparison:
        Which districts appear to be less served or better served when
        combining facility presence, emergency capacity/activity, and
        spatial access patterns?

    Q4. Methodological sensitivity:
        To what extent do district results change when the analytical
        definition of access changes?

WHY NOT USE RAW EMERGENCY VOLUME AS THE MAIN COMPONENT?
    The SUSALUD Tabla C1 only captures activity from IPRESS that actually
    provide emergency services. In practice:
        - Most IPRESS never appear in this table because they are basic posts,
          dental offices, laboratories, or other facilities without formal
          emergency services.
        - Only a minority of IPRESS report real emergency activity.
        - Many districts show zero emergency attendances not because they
          have no healthcare access, but because their facilities are basic
          (I-1, I-2) and not structurally designed for emergency care.

    Therefore, using raw emergency volume as a major component would:
        (a) penalize structurally rural/basic districts,
        (b) provide weak discrimination for a large share of districts,
        (c) conflate lack of reporting with lack of service.

SOLUTION
    We use hospital-level IPRESS categories (I-4 and above) as a structural
    proxy for emergency capacity. Observed emergency activity from Tabla C1
    is kept as a complementary signal with lower weight.

INDEX COMPONENTS
    1. Territorial Supply (C_supply):
       Measures general territorial availability of healthcare resources.

    2. Emergency Capacity (C_emergency):
       Measures structural emergency capacity through hospital-level IPRESS.

    3. Spatial Access (C_spatial):
       Measures how close or far populated centers are from the nearest IPRESS.

    4. Observed Activity (C_activity):
       Measures whether there is evidence of actual emergency activity.
       It is used with lower weight due to the partial nature of C1 coverage.

SPECIFICATIONS
    Baseline:
        More structural, less dependent on C1 reporting.

    Alternative:
        More demanding, more spatially strict, and more sensitive to
        observed emergency activity.

CLASSIFICATION
    Districts are classified using quintiles rather than arbitrary fixed
    thresholds. This allows a relative comparison across the national
    distribution.
===============================================================================
"""

import os
import pandas as pd
import numpy as np

from src.utils import minmax_scale


# ===================================================================== #
#  STEP 1: BUILD THE MASTER DISTRICT TABLE
# ===================================================================== #
def build_district_table(
    distritos_clean,
    district_ipress_count,
    district_emergencias,
    district_distances
) -> pd.DataFrame:
    """
    Merge all district-level data into a single analysis table.

    Starts from the full set of districts and left-joins:
        - IPRESS counts
        - Emergency aggregates
        - CCPP-to-IPRESS distance metrics
    """
    df = distritos_clean[["ubigeo", "departamento", "provincia", "distrito", "area_km2"]].copy()

    df = df.merge(district_ipress_count, on="ubigeo", how="left")
    df = df.merge(district_emergencias, on="ubigeo", how="left")
    df = df.merge(district_distances, on="ubigeo", how="left")

    fill_zero = [
        "n_ipress",
        "n_camas",
        "n_con_internamiento",
        "n_ipress_hospitalaria",
        "total_atenciones",
        "total_atendidos",
        "n_ipress_emergencia",
        "n_ccpp",
    ]
    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    print(f"[build_district_table] {len(df)} districts")
    print(f"  With IPRESS:            {(df['n_ipress'] > 0).sum()}")
    print(f"  With hospital IPRESS:   {(df['n_ipress_hospitalaria'] > 0).sum()}")
    print(f"  With C1 activity > 0:   {(df['total_atendidos'] > 0).sum()}")
    print(f"  With distance data:     {df['mean_dist_km'].notna().sum()}")

    return df


# ===================================================================== #
#  STEP 2: CREATE DERIVED VARIABLES
# ===================================================================== #
def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create density, binary, and transformed variables for index construction.
    """
    result = df.copy()

    safe_area = result["area_km2"].replace(0, np.nan)

    result["ipress_por_100km2"] = result["n_ipress"] / (safe_area / 100)
    result["camas_por_100km2"] = result["n_camas"] / (safe_area / 100)
    result["internamiento_por_100km2"] = result["n_con_internamiento"] / (safe_area / 100)
    result["hospitalaria_por_100km2"] = result["n_ipress_hospitalaria"] / (safe_area / 100)

    for col in [
        "ipress_por_100km2",
        "camas_por_100km2",
        "internamiento_por_100km2",
        "hospitalaria_por_100km2",
    ]:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    result["tiene_emergencia_activa"] = (result["n_ipress_emergencia"] > 0).astype(int)
    result["log_atendidos"] = np.log1p(result["total_atendidos"])

    for col in ["mean_dist_km", "median_dist_km", "max_dist_km"]:
        worst = result[col].max()
        result[col] = result[col].fillna(worst)

    result["pct_beyond_10km"] = result["pct_beyond_10km"].fillna(1.0)
    result["pct_beyond_25km"] = result["pct_beyond_25km"].fillna(1.0)

    _print_derived_stats(result)
    return result


def _print_derived_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for key derived variables."""
    print("\n[derived_variables] Key statistics:")
    for col in [
        "ipress_por_100km2",
        "hospitalaria_por_100km2",
        "mean_dist_km",
        "pct_beyond_25km",
        "tiene_emergencia_activa",
    ]:
        s = df[col]
        print(
            f"  {col:>30s}: mean={s.mean():.3f}, "
            f"median={s.median():.3f}, min={s.min():.3f}, max={s.max():.3f}"
        )


# ===================================================================== #
#  STEP 3: NORMALIZATION HELPERS
# ===================================================================== #
def _normalize_positive(series: pd.Series) -> pd.Series:
    """
    Normalize a variable where higher values mean better access.
    Applies log(1+x) before min-max scaling.
    """
    logged = np.log1p(series)
    return minmax_scale(logged)


def _normalize_negative(series: pd.Series) -> pd.Series:
    """
    Normalize a variable where higher values mean worse access.
    Applies log(1+x), min-max scaling, and then inverts the scale.
    """
    logged = np.log1p(series)
    return 1 - minmax_scale(logged)


# ===================================================================== #
#  STEP 4: COMPONENTS — BASELINE
# ===================================================================== #
def compute_supply_baseline(df: pd.DataFrame) -> pd.Series:
    """
    Baseline territorial supply:
        - IPRESS density per 100 km²
        - Bed density per 100 km²
    """
    c_ipress = _normalize_positive(df["ipress_por_100km2"])
    c_camas = _normalize_positive(df["camas_por_100km2"])
    return 0.50 * c_ipress + 0.50 * c_camas


def compute_emergency_capacity_baseline(df: pd.DataFrame) -> pd.Series:
    """
    Baseline emergency capacity:
        - Hospital-level IPRESS density per 100 km²
        - Binary evidence of observed emergency activity
    """
    c_hosp_density = _normalize_positive(df["hospitalaria_por_100km2"])
    c_tiene_emer = df["tiene_emergencia_activa"].astype(float)
    return 0.80 * c_hosp_density + 0.20 * c_tiene_emer


def compute_spatial_access_baseline(df: pd.DataFrame) -> pd.Series:
    """
    Baseline spatial access:
        - Mean distance to nearest IPRESS
        - Share of populated centers beyond 25 km
    """
    c_mean_dist = _normalize_negative(df["mean_dist_km"])
    c_beyond_25 = _normalize_negative(df["pct_beyond_25km"])
    return 0.60 * c_mean_dist + 0.40 * c_beyond_25


# ===================================================================== #
#  STEP 5: COMPONENTS — ALTERNATIVE
# ===================================================================== #
def compute_supply_alternative(df: pd.DataFrame) -> pd.Series:
    """
    Alternative territorial supply:
        - IPRESS density per 100 km²
        - Inpatient facility density per 100 km²
    """
    c_ipress = _normalize_positive(df["ipress_por_100km2"])
    c_intern = _normalize_positive(df["internamiento_por_100km2"])
    return 0.50 * c_ipress + 0.50 * c_intern


def compute_emergency_activity_alternative(df: pd.DataFrame) -> pd.Series:
    """
    Alternative emergency activity:
        - Observed emergency activity from total_atendidos
    """
    return _normalize_positive(df["log_atendidos"])


def compute_spatial_access_alternative(df: pd.DataFrame) -> pd.Series:
    """
    Alternative spatial access:
        - Median distance to nearest IPRESS
        - Share of populated centers beyond 10 km
    """
    c_median_dist = _normalize_negative(df["median_dist_km"])
    c_beyond_10 = _normalize_negative(df["pct_beyond_10km"])
    return 0.60 * c_median_dist + 0.40 * c_beyond_10


# ===================================================================== #
#  STEP 6: COMPOSITE INDICES
# ===================================================================== #
def compute_eai_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline Emergency Access Index (EAI).
    """
    result = df.copy()

    result["c_supply_base"] = compute_supply_baseline(df)
    result["c_emergency_base"] = compute_emergency_capacity_baseline(df)
    result["c_spatial_base"] = compute_spatial_access_baseline(df)
    result["c_activity_base"] = _normalize_positive(df["log_atendidos"])

    result["eai_baseline"] = (
        0.35 * result["c_supply_base"] +
        0.25 * result["c_emergency_base"] +
        0.30 * result["c_spatial_base"] +
        0.10 * result["c_activity_base"]
    )

    result["quintil_baseline"] = pd.qcut(
        result["eai_baseline"],
        q=5,
        labels=[
            "Q1 (Very low)",
            "Q2 (Low)",
            "Q3 (Medium)",
            "Q4 (High)",
            "Q5 (Very high)",
        ],
        duplicates="drop",
    )
    result["rank_baseline"] = result["eai_baseline"].rank(
        ascending=True, method="min"
    ).astype(int)

    _print_index_summary(result, "eai_baseline", "quintil_baseline", "BASELINE")
    return result


def compute_eai_alternative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative Emergency Access Index (EAI).
    """
    result = df.copy()

    result["c_supply_alt"] = compute_supply_alternative(df)
    result["c_activity_alt"] = compute_emergency_activity_alternative(df)
    result["c_spatial_alt"] = compute_spatial_access_alternative(df)
    result["c_emergency_alt"] = compute_emergency_capacity_baseline(df)

    result["eai_alternative"] = (
        0.25 * result["c_supply_alt"] +
        0.20 * result["c_activity_alt"] +
        0.35 * result["c_spatial_alt"] +
        0.20 * result["c_emergency_alt"]
    )

    result["quintil_alternative"] = pd.qcut(
        result["eai_alternative"],
        q=5,
        labels=[
            "Q1 (Very low)",
            "Q2 (Low)",
            "Q3 (Medium)",
            "Q4 (High)",
            "Q5 (Very high)",
        ],
        duplicates="drop",
    )
    result["rank_alternative"] = result["eai_alternative"].rank(
        ascending=True, method="min"
    ).astype(int)

    _print_index_summary(result, "eai_alternative", "quintil_alternative", "ALTERNATIVE")
    return result


def _print_index_summary(df: pd.DataFrame, score_col: str, class_col: str, label: str) -> None:
    """Print a compact summary for one index specification."""
    print(f"\n[EAI {label}]")
    print(
        f"  Score: mean={df[score_col].mean():.3f}, "
        f"median={df[score_col].median():.3f}, "
        f"std={df[score_col].std():.3f}"
    )
    print("  Distribution:")
    print(df[class_col].value_counts().sort_index().to_string())


# ===================================================================== #
#  STEP 7: SENSITIVITY ANALYSIS
# ===================================================================== #
def compare_specifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare baseline and alternative specifications.
    """
    result = df.copy()

    result["rank_change"] = result["rank_baseline"] - result["rank_alternative"]
    result["abs_rank_change"] = result["rank_change"].abs()
    result["eai_diff"] = result["eai_baseline"] - result["eai_alternative"]
    result["quintil_changed"] = (
        result["quintil_baseline"].astype(str) !=
        result["quintil_alternative"].astype(str)
    )

    n_changed = result["quintil_changed"].sum()
    mean_rank_shift = result["abs_rank_change"].mean()
    corr = result[["eai_baseline", "eai_alternative"]].corr().iloc[0, 1]

    top_movers = result.nlargest(10, "abs_rank_change")[
        [
            "distrito",
            "provincia",
            "departamento",
            "eai_baseline",
            "quintil_baseline",
            "eai_alternative",
            "quintil_alternative",
            "rank_change",
        ]
    ]

    print("\n[Sensitivity Analysis]")
    print(f"  Pearson correlation:         {corr:.4f}")
    print(f"  Districts changing quintile: {n_changed} ({100*n_changed/len(result):.1f}%)")
    print(f"  Mean absolute rank shift:    {mean_rank_shift:.1f}")
    print("\n  Top 10 most sensitive districts:")
    print(top_movers.to_string(index=False))

    return result


# ===================================================================== #
#  STEP 8: FULL METRICS PIPELINE
# ===================================================================== #
def run_metrics_pipeline(distritos_clean, geo_results: dict, save_dir: str = None) -> pd.DataFrame:
    """
    Run the complete metrics pipeline.
    """
    print("\n" + "=" * 70)
    print("  METRICS PIPELINE")
    print("=" * 70)

    dt = build_district_table(
        distritos_clean,
        geo_results["district_ipress_count"],
        geo_results["district_emergencias"],
        geo_results["district_distances"],
    )

    dt = create_derived_variables(dt)
    dt = compute_eai_baseline(dt)
    dt = compute_eai_alternative(dt)
    dt = compare_specifications(dt)

    if save_dir:
        _save_results(dt, save_dir)

    return dt


def _save_results(dt: pd.DataFrame, save_dir: str) -> None:
    """Save output tables."""
    os.makedirs(save_dir, exist_ok=True)

    dt.to_csv(os.path.join(save_dir, "district_analysis.csv"), index=False)

    cols_report = [
        "ubigeo",
        "distrito",
        "provincia",
        "departamento",
        "n_ipress",
        "n_ipress_hospitalaria",
        "n_camas",
        "total_atendidos",
        "mean_dist_km",
        "pct_beyond_25km",
        "eai_baseline",
        "quintil_baseline",
        "eai_alternative",
        "quintil_alternative",
        "rank_change",
    ]
    cols_report = [c for c in cols_report if c in dt.columns]

    dt.nsmallest(20, "eai_baseline")[cols_report].to_csv(
        os.path.join(save_dir, "bottom_20_districts.csv"), index=False
    )
    dt.nlargest(20, "eai_baseline")[cols_report].to_csv(
        os.path.join(save_dir, "top_20_districts.csv"), index=False
    )
    dt.nlargest(50, "abs_rank_change")[cols_report].to_csv(
        os.path.join(save_dir, "specification_comparison.csv"), index=False
    )

    print(f"\n[save] Results saved to {save_dir}/")


# ===================================================================== #
#  MAIN — RUN STANDALONE
# ===================================================================== #
if __name__ == "__main__":
    import warnings
    from src.cleaning import run_cleaning_pipeline
    from src.geospatial import run_geospatial_pipeline

    warnings.filterwarnings("ignore")

    print("\n" + "=" * 70)
    print("  METRICS PIPELINE")
    print("=" * 70)

    print("\n[main] Running cleaning pipeline...")
    distritos_clean, ccpp_clean, ipress_clean, emergencias_clean, _ = run_cleaning_pipeline()

    print("\n[main] Running geospatial pipeline...")
    geo = run_geospatial_pipeline()

    print("\n[main] Running metrics pipeline...")
    dt = run_metrics_pipeline(
        distritos_clean,
        geo,
        save_dir="output/tables"
    )

    print("\n" + "=" * 70)
    print("  TOP 10 DISTRICTS (BASELINE)")
    print("=" * 70)
    top = dt.nlargest(10, "eai_baseline")
    print(
        top[
            ["distrito", "provincia", "departamento", "eai_baseline", "quintil_baseline"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("  BOTTOM 10 DISTRICTS (BASELINE)")
    print("=" * 70)
    bottom = dt.nsmallest(10, "eai_baseline")
    print(
        bottom[
            ["distrito", "provincia", "departamento", "eai_baseline", "quintil_baseline"]
        ].to_string(index=False)
    )

    print("\n[main] metrics.py finished successfully.")
