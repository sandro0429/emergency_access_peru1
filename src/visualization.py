"""
visualization.py — Static charts (matplotlib/seaborn) and maps (GeoPandas/Folium).

Chart selection rationale (documented in README):
    1. Choropleth map of EAI: best way to show geographic inequality patterns
    2. Histogram of EAI distribution: shows the shape of access inequality
    3. Scatter: facility density vs distance — reveals the supply-access relationship
    4. Bar chart: top/bottom districts — concrete comparison of extremes
    5. Box plot by department: shows regional variation
    6. Scatter: baseline vs alternative — sensitivity visualization
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Style defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    'muy_bajo': '#d73027',
    'bajo': '#fc8d59',
    'medio': '#fee08b',
    'alto': '#91cf60',
    'muy_alto': '#1a9850',
}
EAI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'eai', ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850'], N=256
)


def save_fig(fig, filename: str, output_dir: str = "output/figures"):
    """Save figure to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ------------------------------------------------------------------ #
#  LOAD EXISTING FILES ONLY
# ------------------------------------------------------------------ #
def load_existing_inputs():
    """
    Load already-created files only.
    This function does NOT rerun data_loader, cleaning, geospatial, or metrics.
    """
    print("\n" + "=" * 60)
    print("  LOADING EXISTING FILES FOR VISUALIZATION")
    print("=" * 60)

    district_table_path = Path("output/tables/district_analysis.csv")

    district_geo_candidates = [
        Path("data/processed/distritos_clean.geojson"),
        Path("data/processed/districts_clean.geojson"),
        Path("data/processed/distritos_clean.gpkg"),
        Path("data/processed/districts_clean.gpkg"),
        Path("data/processed/distritos_clean.shp"),
        Path("data/processed/districts_clean.shp"),
    ]

    if not district_table_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {district_table_path}"
        )

    district_geo_path = None
    for p in district_geo_candidates:
        if p.exists():
            district_geo_path = p
            break

    if district_geo_path is None:
        raise FileNotFoundError(
            "No se encontró el archivo geográfico de distritos en data/processed/"
        )

    district_table = pd.read_csv(district_table_path, dtype={"ubigeo": str})
    district_table["ubigeo"] = district_table["ubigeo"].astype(str).str.zfill(6)

    districts_gdf = gpd.read_file(district_geo_path)

    if "ubigeo" not in districts_gdf.columns:
        if "IDDIST" in districts_gdf.columns:
            districts_gdf["ubigeo"] = districts_gdf["IDDIST"].astype(str).str.zfill(6)
        elif "iddist" in districts_gdf.columns:
            districts_gdf["ubigeo"] = districts_gdf["iddist"].astype(str).str.zfill(6)
        else:
            raise KeyError(
                "No se encontró una columna ubigeo reconocible en el archivo geográfico."
            )
    else:
        districts_gdf["ubigeo"] = districts_gdf["ubigeo"].astype(str).str.zfill(6)

    print(f"  district_table: {district_table.shape}")
    print(f"  districts_gdf:  {districts_gdf.shape}")
    print(f"  district table file: {district_table_path}")
    print(f"  district geometry file: {district_geo_path}")

    return district_table, districts_gdf


# ------------------------------------------------------------------ #
#  1. CHOROPLETH MAP — EAI by district
# ------------------------------------------------------------------ #
def plot_choropleth_eai(districts_gdf, column='eai_baseline',
                         title='Índice de Acceso a Emergencias (EAI)',
                         output_dir="output/figures"):
    """
    Static choropleth map of the Emergency Access Index.

    WHY THIS CHART: A choropleth is the most intuitive way to show
    geographic patterns of inequality. It answers Q1 and Q3 visually.
    A scatter or bar chart couldn't show the spatial clustering of
    underserved areas (e.g., Amazon region).
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    districts_gdf.plot(
        column=column,
        cmap=EAI_CMAP,
        linewidth=0.2,
        edgecolor='gray',
        legend=True,
        legend_kwds={'label': 'EAI Score', 'orientation': 'horizontal',
                     'shrink': 0.6, 'pad': 0.02},
        ax=ax,
        missing_kwds={'color': 'lightgrey', 'label': 'Sin datos'},
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_axis_off()
    return save_fig(fig, f"choropleth_{column}.png", output_dir)


# ------------------------------------------------------------------ #
#  2. HISTOGRAM — EAI distribution
# ------------------------------------------------------------------ #
def plot_eai_histogram(district_table, output_dir="output/figures"):
    """
    Histogram showing the distribution of EAI scores.

    WHY THIS CHART: Reveals the shape of inequality — whether most
    districts cluster at low or high access. A box plot would show
    quartiles but hide the bimodal pattern that emerges (urban vs rural).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data = district_table['eai_baseline'].dropna()

    ax.hist(data, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(data.median(), color='red', linestyle='--', linewidth=2,
               label=f'Mediana: {data.median():.3f}')
    ax.axvline(data.mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Media: {data.mean():.3f}')

    ax.set_xlabel('EAI Score (Baseline)')
    ax.set_ylabel('Número de distritos')
    ax.set_title('Distribución del Índice de Acceso a Emergencias', fontweight='bold')
    ax.legend()
    return save_fig(fig, "eai_histogram.png", output_dir)


# ------------------------------------------------------------------ #
#  3. SCATTER — Facility density vs. mean distance
# ------------------------------------------------------------------ #
def plot_density_vs_distance(district_table, output_dir="output/figures"):
    """
    Scatter plot: facility density vs. mean distance to IPRESS.

    WHY THIS CHART: Exposes the relationship between supply (facilities)
    and access (distance). Districts in the upper-left quadrant have
    facilities but poor spatial coverage. A correlation heatmap would
    show the coefficient but miss the outlier patterns.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    df = district_table.dropna(subset=['mean_dist_km'])

    density = df['n_ipress'] / (df['area_km2'] / 100)

    scatter = ax.scatter(
        density, df['mean_dist_km'],
        c=df['eai_baseline'], cmap=EAI_CMAP,
        s=20, alpha=0.6, edgecolors='gray', linewidth=0.3
    )
    plt.colorbar(scatter, ax=ax, label='EAI Score', shrink=0.8)

    ax.set_xlabel('Densidad de IPRESS (por 100 km²)')
    ax.set_ylabel('Distancia media al IPRESS más cercano (km)')
    ax.set_title('Densidad de instalaciones vs. acceso espacial', fontweight='bold')
    ax.set_xscale('symlog', linthresh=1)
    return save_fig(fig, "density_vs_distance.png", output_dir)


# ------------------------------------------------------------------ #
#  4. BAR CHART — Top/bottom 15 districts
# ------------------------------------------------------------------ #
def plot_top_bottom_districts(district_table, n=15, output_dir="output/figures"):
    """
    Horizontal bar chart of the top and bottom N districts by EAI.

    WHY THIS CHART: Provides concrete, nameable comparisons for Q3.
    A table would give exact numbers but a bar chart makes the gap
    between best and worst immediately visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Bottom N (worst access)
    bottom = district_table.nsmallest(n, 'eai_baseline').copy()
    bottom['label'] = bottom['distrito'] + ' (' + bottom['provincia'] + ')'
    axes[0].barh(range(n), bottom['eai_baseline'], color='#d73027', alpha=0.85)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(bottom['label'], fontsize=9)
    axes[0].set_xlabel('EAI Score')
    axes[0].set_title(f'Bottom {n} — Peor acceso', fontweight='bold', color='#d73027')
    axes[0].invert_yaxis()

    # Top N (best access)
    top = district_table.nlargest(n, 'eai_baseline').copy()
    top['label'] = top['distrito'] + ' (' + top['provincia'] + ')'
    axes[1].barh(range(n), top['eai_baseline'], color='#1a9850', alpha=0.85)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(top['label'], fontsize=9)
    axes[1].set_xlabel('EAI Score')
    axes[1].set_title(f'Top {n} — Mejor acceso', fontweight='bold', color='#1a9850')
    axes[1].invert_yaxis()

    fig.suptitle('Distritos con mayor y menor acceso a emergencias',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return save_fig(fig, "top_bottom_districts.png", output_dir)


# ------------------------------------------------------------------ #
#  5. BOX PLOT — EAI by department
# ------------------------------------------------------------------ #
def plot_eai_by_department(district_table, output_dir="output/figures"):
    """
    Box plot of EAI scores grouped by department.

    WHY THIS CHART: Shows regional variation and identifies entire
    departments with systematically low access. A bar chart of department
    means would hide the within-department variation that this reveals.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    dept_median = district_table.groupby('departamento')['eai_baseline'].median()
    dept_order = dept_median.sort_values().index.tolist()

    sns.boxplot(
        data=district_table, x='departamento', y='eai_baseline',
        order=dept_order, ax=ax, palette='RdYlGn',
        fliersize=2, linewidth=0.8
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('EAI Score')
    ax.set_title('Distribución del EAI por departamento', fontweight='bold')
    ax.axhline(district_table['eai_baseline'].median(), color='red',
               linestyle='--', alpha=0.5, label='Mediana nacional')
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "eai_by_department.png", output_dir)


# ------------------------------------------------------------------ #
#  6. SCATTER — Baseline vs Alternative
# ------------------------------------------------------------------ #
def plot_sensitivity_comparison(district_table, output_dir="output/figures"):
    """
    Scatter plot comparing baseline vs alternative EAI.

    WHY THIS CHART: Directly answers Q4 (sensitivity). Points near
    the 45° line are stable across specifications. Points far from it
    are sensitive to the methodological choice. A table of rank changes
    would be less intuitive than seeing the cloud shape.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        district_table['eai_baseline'],
        district_table['eai_alternative'],
        c=district_table['abs_rank_change'],
        cmap='YlOrRd', s=15, alpha=0.6,
        edgecolors='gray', linewidth=0.2
    )
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='Línea 45°')
    ax.set_xlabel('EAI Baseline')
    ax.set_ylabel('EAI Alternativo')
    ax.set_title('Sensibilidad: Baseline vs. Alternativo', fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    return save_fig(fig, "sensitivity_scatter.png", output_dir)


# ------------------------------------------------------------------ #
#  7. GROUPED BAR — Classification comparison
# ------------------------------------------------------------------ #
def plot_classification_comparison(district_table, output_dir="output/figures"):
    """
    Grouped bar comparing quintile distributions between specifications.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    quintiles = ['Q1 (Very low)', 'Q2 (Low)', 'Q3 (Medium)',
                 'Q4 (High)', 'Q5 (Very high)']
    colors = [COLORS['muy_bajo'], COLORS['bajo'], COLORS['medio'],
              COLORS['alto'], COLORS['muy_alto']]

    baseline_counts = district_table['quintil_baseline'].value_counts().reindex(quintiles, fill_value=0)
    alt_counts = district_table['quintil_alternative'].value_counts().reindex(quintiles, fill_value=0)

    x = np.arange(len(quintiles))
    width = 0.35

    ax.bar(x - width/2, baseline_counts, width, label='Baseline', color=colors, alpha=0.8,
           edgecolor='white')
    ax.bar(x + width/2, alt_counts, width, label='Alternative', color=colors, alpha=0.5,
           edgecolor='white', hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\nVery low', 'Q2\nLow', 'Q3\nMedium', 'Q4\nHigh', 'Q5\nVery high'])
    ax.set_ylabel('Number of districts')
    ax.set_title('Quintile distribution: Baseline vs Alternative', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "classification_comparison.png", output_dir)


# ------------------------------------------------------------------ #
#  GENERATE ALL STATIC VISUALIZATIONS
# ------------------------------------------------------------------ #
def generate_all_charts(district_table, districts_gdf, output_dir="output/figures"):
    """Generate all static visualizations."""
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    gdf = districts_gdf.merge(
        district_table[['ubigeo', 'eai_baseline', 'eai_alternative',
                        'quintil_baseline', 'quintil_alternative']],
        on='ubigeo', how='left'
    )

    paths = {}
    paths['choropleth_baseline'] = plot_choropleth_eai(
        gdf, 'eai_baseline', 'EAI Baseline — Acceso a Emergencias', output_dir)
    paths['choropleth_alternative'] = plot_choropleth_eai(
        gdf, 'eai_alternative', 'EAI Alternative — Acceso a Emergencias', output_dir)
    paths['histogram'] = plot_eai_histogram(district_table, output_dir)
    paths['density_vs_distance'] = plot_density_vs_distance(district_table, output_dir)
    paths['top_bottom'] = plot_top_bottom_districts(district_table, 15, output_dir)
    paths['by_department'] = plot_eai_by_department(district_table, output_dir)
    paths['sensitivity'] = plot_sensitivity_comparison(district_table, output_dir)
    paths['classification'] = plot_classification_comparison(district_table, output_dir)

    print(f"\n  All charts saved to {output_dir}/")
    return paths


# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VISUALIZATION PIPELINE")
    print("=" * 60)

    district_table, districts_gdf = load_existing_inputs()

    generate_all_charts(district_table, districts_gdf, output_dir="output/figures")

    print("\n[main] visualization.py finished successfully.")
