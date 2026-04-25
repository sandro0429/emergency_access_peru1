# 🏥 Emergency Access Peru — Análisis de Desigualdad en el Acceso a Atención Médica de Emergencia

## ¿Qué hace el proyecto?

Este proyecto analiza la **desigualdad en el acceso a la atención médica de emergencia** a nivel distrital en Perú. Combina cuatro fuentes de datos públicas para construir un Índice de Acceso a Emergencias (EAI) que permite comparar los 1,873 distritos del país.

## Objetivo analítico principal

Responder: **¿Qué distritos de Perú parecen estar mejor o peor atendidos en cuanto al acceso a la atención médica de emergencia, y qué evidencia respalda esa conclusión?**

## Conjuntos de datos utilizados

| Dataset | Registros | Fuente | Formato |
|---------|-----------|--------|---------|
| Límites distritales | 1,873 polígonos | INEI / GeoGPS Perú | Shapefile |
| Centros poblados | 136,587 puntos | IGN / Datos Abiertos | Shapefile |
| Instalaciones IPRESS | 20,793 establecimientos | MINSA / SUSALUD | CSV |
| Producción en Emergencia (Tabla C1) | 323,288 registros | SUSALUD 2025 | CSV |

## Limpieza de datos

### Decisiones clave documentadas

1. **UBIGEO:** Estandarizado a 6 dígitos con ceros a la izquierda en todos los datasets. Necesario porque algunos almacenan el código como entero (ej. 60101 en vez de 060101).

2. **Coordenadas IPRESS — Campos invertidos:** Los campos `NORTE` y `ESTE` del CSV original están intercambiados. `NORTE` contiene longitud (X) y `ESTE` contiene latitud (Y). Esto se verificó con ubicaciones conocidas (ej. Cajamarca: lon≈-78.8, lat≈-6.1). Se corrigió en el pipeline de limpieza.

3. **Coordenadas faltantes:** El 62% de las IPRESS no tiene coordenadas geográficas. Estas instalaciones se incluyen en conteos por distrito (vía UBIGEO) pero no participan en cálculos de distancia, lo que puede subestimar la oferta espacial real.

4. **Validación geográfica:** 3 IPRESS tenían coordenadas fuera del bounding box de Perú (-81.5 a -68.5 lon, -18.4 a 0.1 lat) y fueron excluidas del análisis espacial.

5. **Duplicados:** 26 IPRESS duplicadas por código único y 19,465 registros de emergencia exactamente duplicados fueron eliminados.

6. **CRS de centros poblados:** El shapefile original no tenía CRS definido. Se asignó EPSG:4326 basado en el rango de coordenadas (-81 a -69 lon, -18 a 0 lat).

## Sistema de Referencia de Coordenadas (CRS)

- **EPSG:4326 (WGS 84):** Utilizado para almacenamiento y visualización. Todos los datos se normalizan a este CRS.
- **EPSG:32718 (UTM Zona 18S):** Utilizado para cálculos de distancia (metros) y área (km²). Cubre la mayor parte del territorio peruano.

## Métricas a nivel de distrito

## Metodología del Índice de Acceso a Emergencias (IAE)
- **Lógica general**

El proyecto construye un Índice de Acceso a Emergencias (IAE) a nivel distrital.

La lógica del índice busca capturar cuatro dimensiones complementarias del acceso:

Oferta territorial
Capacidad estructural de emergencia
Acceso espacial
Actividad observada de emergencia

El objetivo no es medir solo la existencia de establecimientos, sino aproximar qué tan bien atendido está un distrito combinando infraestructura, capacidad potencial de emergencia, evidencia observada de uso y distancia desde los centros poblados.

### Índice de Acceso a Emergencias (EAI)

El EAI combina tres componentes normalizados [0, 1]:

**Componente 1 — Disponibilidad de instalaciones (C_fac):**
- Indicador: IPRESS por 100 km², transformado con log(1+x)
- Justificación: La densidad de instalaciones por área es la medida más básica de disponibilidad. La transformación logarítmica evita que Lima distorsione la escala.

**Componente 2 — Actividad de emergencia (C_emer):**
- Indicador: Total de personas atendidas en emergencia (2025), log-transformado
- Justificación: Que existan instalaciones no garantiza servicio de emergencia activo. Este componente captura la oferta funcional.

**Componente 3 — Acceso espacial (C_access):**
- Indicador: Distancia promedio (km) desde centros poblados al IPRESS más cercano (invertida)
- Método: KD-tree sobre coordenadas UTM 18S
- Justificación: La distancia física al centro de salud más cercano es la barrera más directa para la población.

### Especificación Baseline

```
EAI_baseline = 0.40 × C_fac + 0.30 × C_emer + 0.30 × C_access (usando distancia media)
```

Mayor peso a instalaciones (40%) porque sin instalaciones no hay acceso posible.

### Especificación Alternativa

```
EAI_alt = 0.25 × C_fac + 0.25 × C_emer + 0.50 × C_access (usando distancia mediana)
```

Diferencias:
- **Mediana** en vez de media para distancia → menos sensible a outliers (asentamientos remotos extremos)
- **50% de peso** al acceso espacial → refleja que la distancia es la barrera más directa
- Pesos iguales (25%) para los otros dos componentes

### Clasificación

| Clase | Rango EAI | Interpretación |
|-------|-----------|----------------|
| Muy bajo | < 0.15 | Acceso crítico |
| Bajo | 0.15 – 0.30 | Acceso deficiente |
| Medio | 0.30 – 0.55 | Acceso moderado |
| Alto | 0.55 – 0.75 | Buen acceso |
| Muy alto | ≥ 0.75 | Acceso excelente |

## Visualizaciones — Justificación de selección

| Gráfico | Pregunta que responde | Por qué este y no otro |
|---------|----------------------|------------------------|
| Coropleta EAI | Q1, Q3: patrones geográficos | Muestra clustering espacial de desigualdad; un bar chart no capturaría la geografía |
| Histograma EAI | Distribución general | Revela sesgo y bimodalidad; un box plot ocultaría la forma |
| Scatter densidad vs. distancia | Q2: relación oferta-acceso | Dos dimensiones simultáneas con outliers visibles; una correlación perdería estos patrones |
| Barras top/bottom | Q3: distritos extremos | Comparación directa con nombres legibles; comunica la brecha |
| Box plot por departamento | Q1: variación regional | Muestra distribución completa dentro de cada región, no solo la media |
| Scatter baseline vs. alt | Q4: sensibilidad | Visualiza coherencia y divergencia; puntos lejos de la diagonal son sensibles |
| Barras clasificación | Q4: cambio de clases | Comparación directa de distribuciones entre especificaciones |

## Principales hallazgos

1. **Desigualdad extrema:** Los distritos mejor servidos (Lima, Callao, Arequipa) tienen EAI > 0.9, mientras que distritos amazónicos y de sierra profunda tienen EAI < 0.1 — una brecha de 10x.

2. **Patrón geográfico claro:** La costa y las capitales departamentales concentran el acceso. La Amazonía (Loreto, Ucayali) y la sierra sur (Condesuyos, Chucuito) son las zonas más desatendidas.

3. **La distancia importa:** La distancia media al IPRESS más cercano varía de <1 km en Lima a >100 km en distritos fronterizos, siendo la barrera más determinante.

4. **Sensibilidad moderada:** Las dos especificaciones tienen correlación de 0.978, pero el 61% de distritos cambia de categoría — principalmente entre "Bajo" y "Medio", lo que sugiere que muchos distritos están en el umbral.

## Principales limitaciones

1. Distancia euclidiana ≠ tiempo de viaje real (no considera carreteras, ríos, topografía)
2. Sin datos de población → no se pueden calcular tasas per cápita
3. Datos de emergencia con cobertura parcial (63% de distritos)
4. Análisis transversal (no longitudinal)

## Instalación

```bash
1. Clona o descarga este repositorio.
2. Abre una terminal en la carpeta raíz del proyecto.
3. Instala las dependencias con:
pip install -r requirements.txt
```

## Cómo ejecutar el pipeline

```python
# Desde el directorio raíz del proyecto
python3 -c "
from src.data_loader import load_all
from src.cleaning import run_cleaning_pipeline
from src.geospatial import run_geospatial_pipeline
from src.metrics import run_metrics_pipeline
from src.visualization import generate_all_charts

d, c, i, e = load_all('data/raw')
d_c, c_c, i_c, e_c = run_cleaning_pipeline(d, c, i, e, save_dir='data/processed')
geo = run_geospatial_pipeline(d_c, c_c, i_c, e_c)
dt = run_metrics_pipeline(d_c, geo, save_dir='output/tables')
generate_all_charts(dt, d_c, output_dir='output/figures')
"
```

## Cómo ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación tiene 4 pestañas:
1. **Datos y Metodología** — Fuentes, limpieza, definición del EAI
2. **Análisis Estático** — Gráficos con interpretaciones
3. **Resultados Geoespaciales** — Mapas coropléticos y tabla de comparación
4. **Exploración Interactiva** — Mapas Folium y comparador de distritos

## Estructura del repositorio

```
emergency_access_peru1/
├── app.py                    # Streamlit app (4 pestañas)
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias
├── src/
│   ├── data_loader.py        # Carga de los 4 datasets
│   ├── cleaning.py           # Limpieza y preprocesamiento
│   ├── geospatial.py         # Spatial joins, distancias, GeoDataFrames
│   ├── metrics.py            # Índice EAI, baseline y alternativo
│   ├── visualization.py      # Gráficos estáticos (matplotlib/seaborn)
│   └── utils.py              # Funciones auxiliares
├── data/
│   ├── raw/                  # Datos originales descargados
│   └── processed/            # Datos limpios
├── output/
│   ├── figures/              # Gráficos estáticos (PNG)
│   └── tables/               # Tablas de resultados (CSV)
└── video/
    └── link.txt              # Enlace al video explicativo
```

## Herramientas utilizadas

- **GeoPandas:** Manejo de datos geoespaciales, spatial joins, CRS
- **Folium:** Mapas interactivos con tooltips
- **Matplotlib:** Gráficos estáticos (coropléticos, scatter, histogramas)
- **Seaborn:** Box plots estilizados
- **Streamlit:** Aplicación web interactiva
- **SciPy (cKDTree):** Cálculo eficiente de distancias nearest-neighbor
