"""Perform exploratory data analysis on the Online Retail SQLite database."""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_FILE = PROJECT_ROOT / "database" / "online_retail.db"
TABLE_NAME = "transactions"
REPORT_FILE = PROJECT_ROOT / "reports" / "eda_report.txt"

MEANINGFUL_NUMERIC_COLUMNS = {
    "Quantity",
    "UnitPrice",
    "TotalAmount",
    "Hour",
}

BOOLEAN_COLUMNS = [
    "IsCancellation",
    "HasCustomerID",
    "HasDescription",
    "IsZeroPrice",
    "IsNegativePrice",
    "IsNegativeQuantity",
]


def load_from_database() -> pd.DataFrame:
    """Load the transaction table from SQLite into a pandas DataFrame."""
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

    # Restore semantic pandas dtypes after the SQLite round trip.
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    for column in BOOLEAN_COLUMNS:
        df[column] = df[column].astype(bool)

    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the DataFrame reconstructed from SQLite."""
    assert len(df) == 536_641
    assert df.shape[1] == 22

    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert str(df["CustomerID"].dtype) == "Int64"

    for column in BOOLEAN_COLUMNS:
        assert df[column].dtype == bool

    assert df["InvoiceNo"].isna().sum() == 0
    assert df["StockCode"].isna().sum() == 0
    assert df["InvoiceDate"].isna().sum() == 0

    print("DataFrame validation: PASSED")



def build_field_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build reproducible metadata for every field in the dataset."""
    records = []

    for column in df.columns:
        series = df[column]

        record = {
            "Field": column,
            "Dtype": str(series.dtype),
            "Rows": len(series),
            "Missing": int(series.isna().sum()),
            "MissingPct": float(series.isna().mean() * 100),
            "Unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series) and not (
            pd.api.types.is_bool_dtype(series)
        ):
            record["Min"] = series.min()
            record["Max"] = series.max()
            record["Mean"] = (
                series.mean()
                if column in MEANINGFUL_NUMERIC_COLUMNS
                else None
            )
        else:
            record["Min"] = None
            record["Max"] = None
            record["Mean"] = None

        records.append(record)

    return pd.DataFrame(records)


def print_field_profile(profile: pd.DataFrame) -> None:
    """Print metadata generated for every dataset field."""
    print()
    print("=== FIELD METADATA ===")

    for _, row in profile.iterrows():
        print()
        print(f"Field: {row['Field']}")
        print(f"  dtype: {row['Dtype']}")
        print(f"  rows: {int(row['Rows']):,}")
        print(
            f"  missing: {int(row['Missing']):,} "
            f"({row['MissingPct']:.2f}%)"
        )
        print(f"  unique: {int(row['Unique']):,}")

        if pd.notna(row["Min"]):
            print(f"  min: {row['Min']}")
            print(f"  max: {row['Max']}")
            if pd.notna(row["Mean"]):
                print(f"  mean: {row['Mean']}")


def build_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the main descriptive tables used in the EDA."""
    transaction_summary = (
        df.groupby("TransactionType", as_index=False)
        .agg(
            Rows=("InvoiceNo", "size"),
            Amount=("TotalAmount", "sum"),
        )
        .sort_values("Rows", ascending=False)
    )

    weekday_summary = (
        df.groupby("Weekday", as_index=False)
        .agg(
            Rows=("InvoiceNo", "size"),
            Invoices=("InvoiceNo", "nunique"),
            Amount=("TotalAmount", "sum"),
        )
        .sort_values("Rows", ascending=False)
    )

    country_summary = (
        df.groupby("Country", as_index=False)
        .agg(
            Rows=("InvoiceNo", "size"),
            Invoices=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
            Amount=("TotalAmount", "sum"),
        )
        .sort_values("Amount", ascending=False)
    )

    return {
        "transaction_summary": transaction_summary,
        "weekday_summary": weekday_summary,
        "country_summary": country_summary,
    }


def write_eda_report(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Write the main EDA findings and metadata to a text report."""
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    transaction_summary = tables["transaction_summary"]
    weekday_summary = tables["weekday_summary"]
    country_summary = tables["country_summary"]

    net_amount = df["TotalAmount"].sum()
    invoice_count = df["InvoiceNo"].nunique()
    product_count = df["StockCode"].nunique()
    customer_count = df["CustomerID"].nunique()
    active_days = df["Date"].nunique()

    uk_rows = int(
        country_summary.loc[
            country_summary["Country"] == "United Kingdom",
            "Rows",
        ].iloc[0]
    )
    uk_share = 100 * uk_rows / len(df)

    missing_customer = int(df["CustomerID"].isna().sum())
    missing_customer_pct = 100 * missing_customer / len(df)

    observed_weekdays = set(df["Weekday"].dropna().unique())
    expected_weekdays = {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
    missing_weekdays = sorted(expected_weekdays - observed_weekdays)

    lines = [
        "ONLINE RETAIL - EXPLORATORY DATA ANALYSIS REPORT",
        "=" * 52,
        "",
        "1. DATASET OVERVIEW",
        f"Rows: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Unique invoices: {invoice_count:,}",
        f"Unique products: {product_count:,}",
        f"Identified customers: {customer_count:,}",
        f"Active dates: {active_days:,}",
        (
            f"Observed period: {df['InvoiceDate'].min()} "
            f"to {df['InvoiceDate'].max()}"
        ),
        f"Net observed amount: {net_amount:,.2f}",
        "",
        "Interpretation:",
        (
            "The unit of observation is an invoice line, not necessarily "
            "a complete customer purchase. InvoiceNo must therefore be used "
            "when analyses require invoice-level behavior."
        ),
        "",
        "2. DATA QUALITY",
        (
            f"Missing CustomerID: {missing_customer:,} "
            f"({missing_customer_pct:.2f}%)."
        ),
        (
            f"Missing Description: {df['Description'].isna().sum():,} "
            f"({100 * df['Description'].isna().mean():.2f}%)."
        ),
        f"Exact duplicates after cleaning: {df.duplicated().sum():,}.",
        "",
        "Interpretation:",
        (
            "Missing customer identifiers are retained because removing "
            "them would discard a substantial portion of the dataset and "
            "customer identity is not required for every analysis."
        ),
        "",
        "3. FIELD-BY-FIELD METADATA",
        profile.to_string(index=False),
        "",
        "4. TRANSACTION STRUCTURE",
        transaction_summary.to_string(index=False),
        "",
        "Interpretation:",
        (
            "Sales dominate the dataset, while cancellations and other "
            "records are preserved because they contribute to the observed "
            "net monetary result."
        ),
        "",
        "5. GEOGRAPHIC STRUCTURE",
        country_summary.head(10).to_string(index=False),
        "",
        (
            f"United Kingdom rows: {uk_rows:,} "
            f"({uk_share:.2f}% of all rows)."
        ),
        "",
        "Interpretation:",
        (
            "The dataset is strongly concentrated in the United Kingdom. "
            "Aggregate conclusions therefore largely reflect UK activity, "
            "and cross-country comparisons should account for this imbalance."
        ),
        "",
        "6. TEMPORAL STRUCTURE",
        weekday_summary.to_string(index=False),
        "",
        (
            "Weekdays absent from the observed transactions: "
            + (
                ", ".join(missing_weekdays)
                if missing_weekdays
                else "None"
            )
        ),
        "",
        "Interpretation:",
        (
            "Activity is not uniformly distributed across the calendar. "
            "Missing or inactive weekdays must be considered explicitly "
            "before constructing daily time-series or arrival-process models."
        ),
        "",
        "7. MAIN FINDINGS",
        (
            f"- The cleaned analytical dataset contains {len(df):,} "
            "invoice-line observations."
        ),
        f"- There are {invoice_count:,} unique invoices.",
        f"- There are {product_count:,} unique stock codes.",
        f"- There are {customer_count:,} identified customers.",
        (
            f"- CustomerID is missing in {missing_customer_pct:.2f}% "
            "of observations."
        ),
        (
            f"- The United Kingdom represents {uk_share:.2f}% "
            "of all invoice-line observations."
        ),
        (
            f"- The net observed monetary amount is "
            f"{net_amount:,.2f}."
        ),
        (
            "- Extreme positive and negative quantities and amounts are "
            "retained because the audit identified cancellations and "
            "business adjustments rather than treating all extremes as "
            "automatic statistical outliers."
        ),
        "",
        "8. GENERATED METADATA",
        (
            "For each field the analysis generates its pandas dtype, "
            "number of rows, missing-value count and percentage, number "
            "of unique values, and numerical range/mean when meaningful."
        ),
        (
            "Additional dataset-level metadata include temporal coverage, "
            "active dates, invoice count, product count, customer count, "
            "transaction categories, geographic concentration, and net "
            "monetary amount."
        ),
        "",
        "9. QUESTIONS FOR FURTHER ANALYSIS",
        (
            "- How are invoice arrivals distributed through time, and is "
            "a Poisson-type arrival model empirically plausible?"
        ),
        (
            "- Do transaction values exhibit heavy tails or other "
            "non-Gaussian behavior?"
        ),
        (
            "- Is there temporal dependence or seasonality in daily sales?"
        ),
        (
            "- Can daily activity be represented by low, medium, and high "
            "states with meaningful Markov transition probabilities?"
        ),
        (
            "- If temporal dependence and seasonality are supported by the "
            "data, does a SARIMA-type model provide a useful description?"
        ),
    ]

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"EDA report saved: {REPORT_FILE}")

def main() -> None:
    df = load_from_database()

    print("=== EXPLORATORY DATA ANALYSIS ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(
        f"Date range: {df['InvoiceDate'].min()} "
        f"-> {df['InvoiceDate'].max()}"
    )

    validate_dataframe(df)

    profile = build_field_profile(df)
    tables = build_eda_tables(df)
    print_field_profile(profile)
    write_eda_report(df, profile, tables)

    print()
    print("=== PANDAS DTYPES ===")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()









