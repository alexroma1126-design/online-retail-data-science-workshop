from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "database" / "online_retail.db"
TABLE_NAME = "transactions"
RFM_REPORT_FILE = PROJECT_ROOT / "reports" / "rfm_clustering_report.txt"


def load_sales() -> pd.DataFrame:
    """Load identified sale transactions from SQLite."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_FILE}"
        )

    query = f"""
        SELECT
            InvoiceNo,
            InvoiceDate,
            CustomerID,
            TotalAmount
        FROM {TABLE_NAME}
        WHERE
            TransactionType = 'Sale'
            AND CustomerID IS NOT NULL
    """

    with sqlite3.connect(DATABASE_FILE) as connection:
        df = pd.read_sql_query(
            query,
            connection,
            parse_dates=["InvoiceDate"],
        )

    df["CustomerID"] = df["CustomerID"].astype("Int64")

    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Build sale-based customer RFM features."""

    reference_date = (
        df["InvoiceDate"].max().normalize()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
    )

    rfm["Recency"] = (
        reference_date
        - rfm["LastPurchase"].dt.normalize()
    ).dt.days

    rfm = (
        rfm[
            ["Recency", "Frequency", "Monetary"]
        ]
        .reset_index()
        .sort_values("CustomerID")
        .reset_index(drop=True)
    )

    return rfm


def transform_rfm(
    rfm: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Log-transform skewed RFM variables and standardize them."""

    transformed = rfm.copy()

    transformed["LogRecency"] = np.log1p(
        transformed["Recency"]
    )

    transformed["LogFrequency"] = np.log1p(
        transformed["Frequency"]
    )

    transformed["LogMonetary"] = np.log1p(
        transformed["Monetary"]
    )

    feature_columns = [
        "LogRecency",
        "LogFrequency",
        "LogMonetary",
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        transformed[feature_columns]
    )

    return transformed, scaled


def validate_rfm(
    rfm: pd.DataFrame,
    scaled: np.ndarray,
) -> None:
    """Validate customer-level features before clustering."""

    assert rfm["CustomerID"].is_unique
    assert rfm["CustomerID"].notna().all()

    assert (rfm["Recency"] >= 1).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()

    assert scaled.shape == (len(rfm), 3)
    assert np.isfinite(scaled).all()

    means = scaled.mean(axis=0)
    stds = scaled.std(axis=0)

    assert np.allclose(
        means,
        0.0,
        atol=1e-10,
    )

    assert np.allclose(
        stds,
        1.0,
        atol=1e-10,
    )



def evaluate_kmeans(
    scaled: np.ndarray,
    k_values=range(2, 11),
) -> pd.DataFrame:
    """Evaluate K-Means solutions using inertia and silhouette score."""

    results = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(scaled)

        results.append(
            {
                "k": k,
                "Inertia": model.inertia_,
                "Silhouette": silhouette_score(
                    scaled,
                    labels,
                ),
            }
        )

    return pd.DataFrame(results)


def profile_kmeans(
    rfm: pd.DataFrame,
    scaled: np.ndarray,
    n_clusters: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit K-Means and summarize the resulting customer segments."""

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(scaled)

    clustered = rfm.copy()
    clustered["Cluster"] = labels

    profile = (
        clustered.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            RecencyMedian=("Recency", "median"),
            RecencyMean=("Recency", "mean"),
            FrequencyMedian=("Frequency", "median"),
            FrequencyMean=("Frequency", "mean"),
            MonetaryMedian=("Monetary", "median"),
            MonetaryMean=("Monetary", "mean"),
        )
        .reset_index()
    )

    profile["CustomerShare"] = (
        profile["Customers"]
        / len(clustered)
        * 100
    )

    return clustered, profile


def plot_clusters_pca(
    clustered: pd.DataFrame,
    scaled: np.ndarray,
) -> tuple[Path, np.ndarray, pd.DataFrame]:
    """Project standardized RFM data to two PCA dimensions."""

    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(scaled)

    explained = pca.explained_variance_ratio_

    loadings = pd.DataFrame(
        pca.components_.T,
        index=[
            "LogRecency",
            "LogFrequency",
            "LogMonetary",
        ],
        columns=[
            "PC1",
            "PC2",
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=clustered["Cluster"],
        s=18,
        alpha=0.55,
    )

    ax.set_title(
        "Segmentación RFM proyectada mediante PCA"
    )

    ax.set_xlabel(
        f"Componente principal 1 ({explained[0] * 100:.1f}% varianza)"
    )

    ax.set_ylabel(
        f"Componente principal 2 ({explained[1] * 100:.1f}% varianza)"
    )

    ax.grid(alpha=0.2)

    fig.colorbar(
        scatter,
        ax=ax,
        label="Cluster",
    )

    output_file = (
        PROJECT_ROOT
        / "figures"
        / "04_rfm_clusters_pca.png"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file, explained, loadings


def write_rfm_report(
    rfm: pd.DataFrame,
    k_results: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    explained_variance: np.ndarray,
    pca_loadings: pd.DataFrame,
    pca_figure: Path,
) -> Path:
    """Write a reproducible Spanish report of the RFM clustering analysis."""

    RFM_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_row = k_results.loc[
        k_results["Silhouette"].idxmax()
    ]

    best_k = int(best_row["k"])
    best_silhouette = float(
        best_row["Silhouette"]
    )

    active_row = cluster_profile.loc[
        cluster_profile["RecencyMedian"].idxmin()
    ]

    less_active_row = cluster_profile.loc[
        cluster_profile["RecencyMedian"].idxmax()
    ]

    comparison = k_results.to_string(
        index=False,
        formatters={
            "Inertia": lambda x: f"{x:,.2f}",
            "Silhouette": lambda x: f"{x:.4f}",
        },
    )

    profile_text = cluster_profile.to_string(
        index=False,
        formatters={
            "CustomerShare": lambda x: f"{x:.2f}%",
            "RecencyMedian": lambda x: f"{x:,.1f}",
            "RecencyMean": lambda x: f"{x:,.1f}",
            "FrequencyMedian": lambda x: f"{x:,.1f}",
            "FrequencyMean": lambda x: f"{x:,.2f}",
            "MonetaryMedian": lambda x: f"{x:,.2f}",
            "MonetaryMean": lambda x: f"{x:,.2f}",
        },
    )

    loadings_text = pca_loadings.to_string(
        float_format=lambda x: f"{x:.4f}"
    )

    lines = [
        "ONLINE RETAIL - SEGMENTACI\u00d3N RFM",
        "=" * 40,
        "",
        "1. Objetivo",
        (
            "Explorar si los clientes identificados presentan grupos "
            "diferenciados de comportamiento seg\u00fan recencia, frecuencia "
            "y valor monetario."
        ),
        "",
        "2. Preparaci\u00f3n de datos",
        f"Clientes analizados: {len(rfm):,}.",
        (
            "Se utilizaron exclusivamente transacciones clasificadas como "
            "Sale con CustomerID disponible."
        ),
        (
            "Las variables RFM fueron transformadas mediante log(1+x) y "
            "posteriormente estandarizadas antes de aplicar K-Means."
        ),
        "",
        "3. Comparaci\u00f3n de modelos K-Means",
        comparison,
        "",
        (
            f"El mayor coeficiente silhouette se obtuvo con k={best_k}: "
            f"{best_silhouette:.4f}."
        ),
        (
            "La inercia disminuye al aumentar k, como es esperable en "
            "K-Means, por lo que el silhouette se utiliz\u00f3 como criterio "
            "principal de separaci\u00f3n."
        ),
        "",
        "4. Perfil de la soluci\u00f3n k=2",
        profile_text,
        "",
        (
            "El grupo con comportamiento m\u00e1s activo representa "
            f"{active_row['CustomerShare']:.2f}% de los clientes, "
            f"con recencia mediana de {active_row['RecencyMedian']:.0f} "
            "d\u00edas, frecuencia mediana de "
            f"{active_row['FrequencyMedian']:.0f} y valor monetario "
            f"mediano de {active_row['MonetaryMedian']:,.2f}."
        ),
        (
            "El grupo menos activo representa "
            f"{less_active_row['CustomerShare']:.2f}% de los clientes, "
            f"con recencia mediana de {less_active_row['RecencyMedian']:.0f} "
            "d\u00edas, frecuencia mediana de "
            f"{less_active_row['FrequencyMedian']:.0f} y valor monetario "
            f"mediano de {less_active_row['MonetaryMedian']:,.2f}."
        ),
        "",
        "5. An\u00e1lisis de componentes principales",
        (
            f"PC1 explica {explained_variance[0] * 100:.2f}% "
            "de la varianza."
        ),
        (
            f"PC2 explica {explained_variance[1] * 100:.2f}% "
            "de la varianza."
        ),
        (
            f"Las dos primeras componentes explican conjuntamente "
            f"{explained_variance.sum() * 100:.2f}% de la varianza."
        ),
        "",
        "Cargas PCA:",
        loadings_text,
        "",
        (
            "PC1 contrapone principalmente la recencia frente a la "
            "frecuencia y el valor monetario, por lo que resume una "
            "dimensi\u00f3n general de intensidad de la relaci\u00f3n comercial."
        ),
        (
            "PC2 est\u00e1 dominada principalmente por la recencia y captura "
            "una segunda fuente de variabilidad temporal entre clientes."
        ),
        "",
        f"Figura PCA: {pca_figure.name}",
        "",
        "6. Conclusi\u00f3n",
        (
            "Entre las soluciones K-Means evaluadas, la partici\u00f3n en "
            "dos grupos proporciona la separaci\u00f3n m\u00e1s clara seg\u00fan "
            "el coeficiente silhouette."
        ),
        (
            "Los grupos obtenidos presentan diferencias marcadas y "
            "coherentes en recencia, frecuencia y valor monetario."
        ),
        (
            "El PCA conserva el 93.86% de la variabilidad en dos dimensiones, "
            "lo que permite representar visualmente gran parte de la "
            "estructura del espacio RFM."
        ),
        "",
        (
            "Estos resultados respaldan la existencia de heterogeneidad "
            "estructurada en el comportamiento de compra de los clientes."
        ),
    ]

    RFM_REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return RFM_REPORT_FILE


def main() -> None:
    sales = load_sales()
    rfm = build_rfm(sales)
    transformed, scaled = transform_rfm(rfm)

    validate_rfm(
        rfm,
        scaled,
    )

    k_results = evaluate_kmeans(scaled)

    clustered, cluster_profile = profile_kmeans(
        rfm,
        scaled,
        n_clusters=2,
    )

    pca_figure, explained_variance, pca_loadings = plot_clusters_pca(
        clustered,
        scaled,
    )

    rfm_report = write_rfm_report(
        rfm,
        k_results,
        cluster_profile,
        explained_variance,
        pca_loadings,
        pca_figure,
    )

    print("=== RFM CLUSTERING DATA ===")
    print(f"Sale rows: {len(sales):,}")
    print(f"Customers: {len(rfm):,}")
    print(
        f"Reference date: "
        f"{sales['InvoiceDate'].max().normalize().date() + pd.Timedelta(days=1)}"
    )

    print()
    print("=== ORIGINAL RFM ===")
    print(
        rfm[
            ["Recency", "Frequency", "Monetary"]
        ].describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("=== TRANSFORMED FEATURES ===")
    print(
        transformed[
            [
                "LogRecency",
                "LogFrequency",
                "LogMonetary",
            ]
        ].describe().to_string()
    )

    print()
    print("=== STANDARDIZED FEATURES ===")
    print(
        "Means:",
        np.round(
            scaled.mean(axis=0),
            6,
        ),
    )

    print(
        "Standard deviations:",
        np.round(
            scaled.std(axis=0),
            6,
        ),
    )

    print()
    print("RFM validation: PASSED")

    print()
    print("=== K-MEANS MODEL COMPARISON ===")
    print(
        k_results.to_string(
            index=False,
            formatters={
                "Inertia": lambda x: f"{x:,.2f}",
                "Silhouette": lambda x: f"{x:.4f}",
            },
        )
    )

    print()
    print("=== K=2 CLUSTER PROFILE ===")
    print(
        cluster_profile.to_string(
            index=False,
            formatters={
                "CustomerShare": lambda x: f"{x:.2f}%",
                "RecencyMedian": lambda x: f"{x:,.1f}",
                "RecencyMean": lambda x: f"{x:,.1f}",
                "FrequencyMedian": lambda x: f"{x:,.1f}",
                "FrequencyMean": lambda x: f"{x:,.2f}",
                "MonetaryMedian": lambda x: f"{x:,.2f}",
                "MonetaryMean": lambda x: f"{x:,.2f}",
            },
        )
    )



    print()
    print("=== PCA ===")
    print(
        f"PC1 explained variance: "
        f"{explained_variance[0] * 100:.2f}%"
    )
    print(
        f"PC2 explained variance: "
        f"{explained_variance[1] * 100:.2f}%"
    )
    print(
        f"Total explained variance: "
        f"{explained_variance.sum() * 100:.2f}%"
    )
    print(
        f"PCA figure saved: {pca_figure}"
    )
    print(
        f"RFM report saved: {rfm_report}"
    )

    print()
    print("=== PCA LOADINGS ===")
    print(
        pca_loadings.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

if __name__ == "__main__":
    main()













