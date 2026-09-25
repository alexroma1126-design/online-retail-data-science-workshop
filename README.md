# Taller 1 — Adquisición, procesamiento y visualización de datos

**Maestría en Ciencia de Datos — Yachay Tech**  
**Asignatura:** Fundamentos de Ciencia de Datos

## Descripción

Este proyecto desarrolla un flujo reproducible para la adquisición, limpieza, almacenamiento, análisis exploratorio y visualización del conjunto de datos **Online Retail** del UCI Machine Learning Repository.

El proceso implementado sigue cuatro etapas principales:

1. limpieza y preparación de los datos;
2. almacenamiento en SQLite;
3. análisis exploratorio de datos;
4. generación de visualizaciones e interpretación.

Los scripts fueron diseñados para que cada etapa pueda ejecutarse nuevamente desde la terminal y valide automáticamente sus resultados.

---

## Conjunto de datos

El conjunto original corresponde a transacciones de una empresa minorista en línea entre diciembre de 2010 y diciembre de 2011.

Variables originales principales:

- `InvoiceNo`: número de factura;
- `StockCode`: código del producto;
- `Description`: descripción del producto;
- `Quantity`: cantidad;
- `InvoiceDate`: fecha y hora de la transacción;
- `UnitPrice`: precio unitario;
- `CustomerID`: identificador del cliente;
- `Country`: país.

El archivo original contiene:

- 541,909 registros;
- 8 columnas;
- 5,268 registros exactamente duplicados.

---

## Estructura del proyecto

    Taller 1/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── database/
    ├── figures/
    │   ├── 01_daily_sales.png
    │   ├── 02_rfm_customers.png
    │   └── 03_product_cooccurrence.png
    ├── reports/
    │   ├── eda_report.txt
    │   └── visualizations_report.txt
    ├── src/
    │   ├── 01_load_clean.py
    │   ├── 02_create_database.py
    │   ├── 03_eda.py
    │   ├── 04_visualizations.py
    │   └── stochastic/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

Los datos procesados y la base SQLite pueden regenerarse mediante los scripts del proyecto.

---

## Preparación del entorno

Se utilizó Python 3.13 dentro de un entorno virtual.

Crear el entorno:

    python -m venv .venv

Activarlo en PowerShell:

    .venv\Scripts\Activate.ps1

Instalar las dependencias:

    pip install -r requirements.txt

---

## 1. Limpieza y preparación

Ejecutar:

    python src/01_load_clean.py

El script:

- carga el archivo original;
- realiza una auditoría de calidad;
- elimina duplicados exactos;
- conserva cancelaciones y ajustes;
- identifica cantidades y precios anómalos;
- conserva registros sin `CustomerID`;
- genera variables auxiliares;
- calcula `TotalAmount = Quantity * UnitPrice`;
- genera variables temporales;
- clasifica las transacciones.

Después de eliminar duplicados exactos se obtienen:

- 536,641 registros;
- 22 columnas.

Tipos de transacción:

- Sale: 524,878;
- Cancellation: 9,251;
- Adjustment: 1,336;
- Other: 1,176.

El archivo procesado se genera en:

    data/processed/online_retail_clean.csv

---

## 2. Base de datos SQLite

Ejecutar:

    python src/02_create_database.py

El script carga los datos procesados en:

    database/online_retail.db

La tabla principal se denomina:

    transactions

Se crean índices sobre:

- `InvoiceNo`;
- `InvoiceDate`;
- `CustomerID`;
- `Country`;
- `TransactionType`.

La validación final comprueba que la base contenga los mismos 536,641 registros del conjunto procesado.

---

## 3. Análisis exploratorio de datos

Ejecutar:

    python src/03_eda.py

El análisis se realiza leyendo directamente desde SQLite y transformando la tabla nuevamente en un `DataFrame` de pandas.

Se estudian:

- dimensiones del conjunto;
- tipos de datos;
- valores faltantes;
- valores únicos;
- estadísticas descriptivas;
- estructura de las transacciones;
- distribución geográfica;
- comportamiento temporal;
- metadatos de cada variable.

Resultados generales:

- 25,900 facturas;
- 4,070 productos;
- 4,372 clientes identificados;
- 38 países;
- 305 días con actividad.

El reporte completo se genera en:

    reports/eda_report.txt

---

## 4. Visualización de datos

Ejecutar:

    python src/04_visualizations.py

El script genera tres visualizaciones principales.

### Figura 1 — Ventas netas diarias

Archivo:

    figures/01_daily_sales.png

Presenta las ventas netas por día junto con un promedio móvil de siete días.

Permite observar:

- variabilidad diaria;
- tendencia temporal;
- patrones semanales;
- días sin actividad.

### Figura 2 — Estructura RFM de clientes

Archivo:

    figures/02_rfm_customers.png

Representa conjuntamente:

- recencia;
- frecuencia;
- valor monetario.

Las transformaciones logarítmicas permiten visualizar mejor la fuerte asimetría existente entre clientes.

Esta estructura constituye una base para posteriores técnicas de segmentación.

### Figura 3 — Co-compra de productos

Archivo:

    figures/03_product_cooccurrence.png

Se construye una matriz de coocurrencia para los productos más frecuentes.

La visualización permite estudiar qué productos aparecen repetidamente dentro de las mismas facturas y constituye una base para:

- análisis de cesta de mercado;
- sistemas de recomendación;
- teoría de grafos;
- detección de comunidades.

El análisis textual de las figuras se genera en:

    reports/visualizations_report.txt

---

## Ejecución completa

Con el archivo original disponible, el flujo completo puede reproducirse ejecutando:

    python src/01_load_clean.py
    python src/02_create_database.py
    python src/03_eda.py
    python src/04_visualizations.py

Cada etapa incorpora validaciones antes de finalizar.

---

## Decisiones metodológicas

La limpieza se diseñó para evitar eliminar información potencialmente válida.

Por esta razón, no se consideran automáticamente errores:

- cantidades negativas;
- precios iguales a cero;
- cancelaciones;
- ajustes administrativos;
- registros sin identificador de cliente.

En lugar de eliminarlos, estos casos se identifican y clasifican para poder decidir posteriormente qué observaciones deben utilizarse según el objetivo específico del análisis.

---

## Reproducibilidad

El proyecto utiliza Git para mantener cada etapa separada y trazable.

El flujo implementado es:

    datos originales
          ↓
    limpieza
          ↓
    datos procesados
          ↓
    SQLite
          ↓
    análisis exploratorio
          ↓
    visualizaciones
          ↓
    reportes

Los archivos derivados pueden regenerarse a partir de los scripts incluidos.

---

## Extensiones avanzadas

La exploración inicial muestra oportunidades para ampliar posteriormente el proyecto mediante:

- segmentación RFM y clustering;
- procesos estocásticos;
- análisis de tiempos entre transacciones;
- series temporales y SARIMA;
- detección de anomalías;
- grafos de co-compra;
- análisis espectral;
- modelos de machine learning.

Estas extensiones no forman parte del flujo base actualmente validado y se desarrollarán de manera independiente.

---

## Estado actual

El flujo base de adquisición, limpieza, persistencia en SQLite, análisis exploratorio y visualización se encuentra implementado y validado.

Las extensiones estadísticas y de machine learning permanecen en desarrollo.
