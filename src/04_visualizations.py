"""Create exploratory visualizations for the Online Retail dataset."""

from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_FILE = PROJECT_ROOT / "database" / "online_retail.db"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_FILE = PROJECT_ROOT / "reports" / "visualizations_report.txt"

TABLE_NAME = "transactions"


def load_data() -> pd.DataFrame:
    """Load analytical data directly from SQLite."""
    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_FILE}"
        )

    with sqlite3.connect(DATABASE_FILE) as connection:
        df = pd.read_sql_query(
            f"SELECT * FROM {TABLE_NAME}",
            connection,
            parse_dates=["InvoiceDate", "Date"],
        )

    return df


def prepare_visualization_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare aggregated datasets used by the visualizations."""

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            NetAmount=("TotalAmount", "sum"),
            Invoices=("InvoiceNo", "nunique"),
        )
        .sort_values("Date")
    )

    countries = (
        df.groupby("Country", as_index=False)
        .agg(
            NetAmount=("TotalAmount", "sum"),
            Invoices=("InvoiceNo", "nunique"),
        )
        .sort_values("NetAmount", ascending=False)
    )

    invoices = (
        df.groupby("InvoiceNo", as_index=False)
        .agg(
            InvoiceAmount=("TotalAmount", "sum"),
            InvoiceDate=("InvoiceDate", "min"),
            Country=("Country", "first"),
        )
    )

    return daily, countries, invoices


def validate_visualization_data(
    daily: pd.DataFrame,
    countries: pd.DataFrame,
    invoices: pd.DataFrame,
) -> None:
    """Validate the aggregated datasets before plotting."""
    assert len(daily) == 305
    assert len(countries) == 38
    assert len(invoices) == 25_900

    print("Visualization data validation: PASSED")




def prepare_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level recency, frequency, and monetary features."""

    identified = df[df["CustomerID"].notna()].copy()

    invoice_level = (
        identified.groupby(
            ["CustomerID", "InvoiceNo"],
            as_index=False,
        )
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            InvoiceAmount=("TotalAmount", "sum"),
        )
    )

    reference_date = df["InvoiceDate"].max().normalize()

    rfm = (
        invoice_level.groupby("CustomerID", as_index=False)
        .agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("InvoiceAmount", "sum"),
            LastPurchase=("InvoiceDate", "max"),
        )
    )

    rfm["Recency"] = (
        reference_date
        - rfm["LastPurchase"].dt.normalize()
    ).dt.days

    return rfm

def plot_daily_sales(daily: pd.DataFrame) -> Path:
    """Plot daily net sales and a seven-day moving average."""

    series = (
        daily.set_index("Date")["NetAmount"]
        .sort_index()
        .reindex(
            pd.date_range(
                daily["Date"].min(),
                daily["Date"].max(),
                freq="D",
            ),
            fill_value=0,
        )
    )

    moving_average = series.rolling(
        window=7,
        min_periods=1,
    ).mean()

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        series.index,
        series.values,
        linewidth=0.9,
        alpha=0.55,
        label="Daily net sales",
    )

    ax.plot(
        moving_average.index,
        moving_average.values,
        linewidth=2.2,
        label="7-day moving average",
    )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_title(
        "Ventas netas diarias y promedio móvil de 7 días"
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Monto de ventas netas")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    output_file = FIGURES_DIR / "01_daily_sales.png"

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file


def plot_rfm_customers(rfm: pd.DataFrame) -> Path:
    """Visualize customer recency, frequency, and monetary behavior."""

    plot_data = rfm.copy()

    plot_data["LogFrequency"] = np.log1p(
        plot_data["Frequency"]
    )

    plot_data["SignedLogMonetary"] = (
        np.sign(plot_data["Monetary"])
        * np.log1p(np.abs(plot_data["Monetary"]))
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        plot_data["Recency"],
        plot_data["LogFrequency"],
        c=plot_data["SignedLogMonetary"],
        s=22,
        alpha=0.55,
    )

    colorbar = fig.colorbar(scatter, ax=ax)

    colorbar.set_label(
        "Valor monetario transformado con logaritmo con signo"
    )

    ax.set_title(
        "Estructura RFM de clientes"
    )
    ax.set_xlabel(
        "Recencia (días desde la última compra)"
    )
    ax.set_ylabel(
        "log(1 + frecuencia de compra)"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    output_file = FIGURES_DIR / "02_rfm_customers.png"

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file


def prepare_product_cooccurrence(
    df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Build a co-occurrence matrix for the most frequent sale products."""

    sales = df[df["TransactionType"] == "Sale"].copy()

    product_frequency = (
        sales.groupby("StockCode")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )

    top_products = product_frequency.head(top_n).index

    selected = (
        sales[sales["StockCode"].isin(top_products)]
        [["InvoiceNo", "StockCode"]]
        .drop_duplicates()
    )

    basket = pd.crosstab(
        selected["InvoiceNo"],
        selected["StockCode"],
    )

    basket = (basket > 0).astype(int)

    cooccurrence = basket.T.dot(basket)

    cooccurrence_array = cooccurrence.to_numpy(copy=True)

    np.fill_diagonal(
        cooccurrence_array,
        0,
    )

    cooccurrence = pd.DataFrame(
        cooccurrence_array,
        index=cooccurrence.index,
        columns=cooccurrence.columns,
    )

    return cooccurrence


def plot_product_cooccurrence(
    cooccurrence: pd.DataFrame,
) -> Path:
    """Plot product co-purchase structure as a heatmap."""

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        cooccurrence.values,
        aspect="auto",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Número de facturas de venta compartidas",
    )

    labels = cooccurrence.index.astype(str)

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
    )

    ax.set_yticklabels(labels)

    ax.set_title(
        "Estructura de co-compra de productos"
    )

    ax.set_xlabel("StockCode")
    ax.set_ylabel("StockCode")

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "03_product_cooccurrence.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file


def write_visualization_report(
    daily: pd.DataFrame,
    countries: pd.DataFrame,
    invoices: pd.DataFrame,
    rfm: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    """Write a short report describing the saved visualizations."""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    daily_calendar = (
        daily.set_index("Date")["NetAmount"]
        .sort_index()
        .reindex(
            pd.date_range(
                daily["Date"].min(),
                daily["Date"].max(),
                freq="D",
            ),
            fill_value=0,
        )
    )

    peak_day = daily.loc[daily["NetAmount"].idxmax()]
    low_day = daily.loc[daily["NetAmount"].idxmin()]

    top_country = countries.iloc[0]
    second_country = countries.iloc[1]

    top_pair_value = cooccurrence.stack().max()

    lines = [
        "ONLINE RETAIL - INFORME DE VISUALIZACIONES",
        "=" * 38,
        "",
        "Figuras guardadas:",
    ]

    for figure_path in figure_paths:
        lines.append(f"- {figure_path.name}")

    lines += [
        "",
        "1. Figure: 01_daily_sales.png",
        "Técnica: gráfico de línea de serie temporal con promedio móvil de 7 días.",
        (
            f"Peak active day: {peak_day['Date'].date()} "
            f"with net sales {peak_day['NetAmount']:,.2f}."
        ),
        (
            f"Lowest active day: {low_day['Date'].date()} "
            f"with net sales {low_day['NetAmount']:,.2f}."
        ),
        (
            f"Calendar days represented: {len(daily_calendar):,}, "
            f"including {(daily_calendar == 0).sum():,} zero-activity days."
        ),
        (
            "Interpretación: daily sales are highly variable, so the moving average "
            "helps reveal the broader weekly pattern beneath the day-to-day noise."
        ),
        "",
        "2. Figure: 02_rfm_customers.png",
        "Technique: scatter plot of Recency vs. log-transformed Frequency, colored by signed log Monetary value.",
        f"Customers represented: {len(rfm):,}.",
        (
            f"Median recency: {rfm['Recency'].median():.0f} days; "
            f"median frequency: {rfm['Frequency'].median():.0f}; "
            f"median monetary value: {rfm['Monetary'].median():,.2f}."
        ),
        (
            "Interpretación: customer behavior is heterogeneous, with clear differences "
            "in time since last purchase, purchasing frequency, and cumulative value."
        ),
        "",
        "3. Figure: 03_product_cooccurrence.png",
        "Técnica: mapa de calor de co-compra para los productos más frecuentes.",
        (
            f"Top country by net amount in the dataset: "
            f"{top_country['Country']} ({top_country['NetAmount']:,.2f})."
        ),
        (
            f"Second country by net amount: "
            f"{second_country['Country']} ({second_country['NetAmount']:,.2f})."
        ),
        (
            f"Largest co-occurrence count in the displayed matrix: "
            f"{top_pair_value:,} shared sale invoices."
        ),
        (
            "Interpretación: products are not purchased independently; the heatmap "
            "reveals repeated co-purchase structure and motivates future "
            "basket-analysis or graph-based modeling."
        ),
        "",
        "Conclusión general:",
        (
            "The three visualizations complement each other: the first focuses "
            "on temporal dynamics, the second on customer behavior, and the "
            "third on relationships among products."
        ),
    ]

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return REPORT_FILE

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    daily, countries, invoices = prepare_visualization_data(df)
    rfm = prepare_rfm(df)
    cooccurrence = prepare_product_cooccurrence(df)

    print("=== VISUALIZATION DATA ===")
    print(f"Source rows: {len(df):,}")
    print(f"Active dates: {len(daily):,}")
    print(f"Countries: {len(countries):,}")
    print(f"Invoices: {len(invoices):,}")

    validate_visualization_data(
        daily,
        countries,
        invoices,
    )

    daily_sales_figure = plot_daily_sales(daily)
    print(f"Daily sales figure saved: {daily_sales_figure}")

    rfm_figure = plot_rfm_customers(rfm)
    print(f"RFM figure saved: {rfm_figure}")

    cooccurrence_figure = plot_product_cooccurrence(
        cooccurrence
    )
    print(
        f"Product co-occurrence figure saved: "
        f"{cooccurrence_figure}"
    )

    visualization_report = write_visualization_report(
        daily,
        countries,
        invoices,
        rfm,
        cooccurrence,
        [
            daily_sales_figure,
            rfm_figure,
            cooccurrence_figure,
        ],
    )
    print(
        f"Visualization report saved: "
        f"{visualization_report}"
    )

    print()
    print("=== RFM DATA ===")
    print(f"Customers: {len(rfm):,}")
    print()
    print(
        rfm[
            ["Recency", "Frequency", "Monetary"]
        ].describe().to_string()
    )

    print()
    print("=== DAILY DATA ===")
    print(daily.head().to_string(index=False))

    print()
    print("=== TOP COUNTRIES ===")
    print(countries.head(10).to_string(index=False))

    print()
    print("=== INVOICE AMOUNTS ===")
    print(invoices["InvoiceAmount"].describe().to_string())


if __name__ == "__main__":
    main()






















