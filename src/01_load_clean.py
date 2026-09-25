"""Load, audit, clean, and enrich the UCI Online Retail dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.csv"


def load_raw_data() -> pd.DataFrame:
    """Load the original Excel dataset without modifying it."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}"
        )

    return pd.read_excel(RAW_FILE)



def audit_data(df: pd.DataFrame) -> dict[str, int]:
    """Compute key data-quality indicators before cleaning."""
    invoice = df["InvoiceNo"].astype(str)
    cancellations = invoice.str.startswith("C")

    audit = {
        "raw_rows": len(df),
        "exact_duplicates": int(df.duplicated().sum()),
        "missing_description": int(df["Description"].isna().sum()),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "cancellations": int(cancellations.sum()),
        "negative_quantity": int((df["Quantity"] < 0).sum()),
        "negative_quantity_non_cancellation": int(
            ((df["Quantity"] < 0) & ~cancellations).sum()
        ),
        "zero_unit_price": int((df["UnitPrice"] == 0).sum()),
        "negative_unit_price": int((df["UnitPrice"] < 0).sum()),
    }

    return audit


def print_audit(audit: dict[str, int]) -> None:
    """Print the audit results in a readable format."""
    print()
    print("=== DATA QUALITY AUDIT ===")

    for name, value in audit.items():
        print(f"{name:40s}: {value:,}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the analytical dataset while preserving business information."""
    clean = df.copy()

    # Exact duplicate rows add no new information.
    clean = clean.drop_duplicates().copy()

    # Normalize text fields without converting missing values to strings.
    clean["Description"] = clean["Description"].astype("string").str.strip()
    clean["Country"] = clean["Country"].astype("string").str.strip()

    # CustomerID is an identifier, not a continuous numerical variable.
    clean["CustomerID"] = clean["CustomerID"].astype("Int64")

    # Business and data-quality flags.
    clean["IsCancellation"] = (
        clean["InvoiceNo"].astype(str).str.startswith("C")
    )
    clean["HasCustomerID"] = clean["CustomerID"].notna()
    clean["HasDescription"] = clean["Description"].notna()
    clean["IsZeroPrice"] = clean["UnitPrice"].eq(0)
    clean["IsNegativePrice"] = clean["UnitPrice"].lt(0)
    clean["IsNegativeQuantity"] = clean["Quantity"].lt(0)

    # Conservative transaction classification.
    # Ambiguous records are preserved instead of being silently discarded.
    clean["TransactionType"] = "Other"

    sale_mask = (
        (clean["Quantity"] > 0)
        & (clean["UnitPrice"] > 0)
        & ~clean["IsCancellation"]
    )
    adjustment_mask = (
        (clean["Quantity"] < 0)
        & ~clean["IsCancellation"]
    )

    clean.loc[sale_mask, "TransactionType"] = "Sale"
    clean.loc[adjustment_mask, "TransactionType"] = "Adjustment"
    clean.loc[clean["IsCancellation"], "TransactionType"] = "Cancellation"

    # Monetary contribution of each row.
    # Negative quantities therefore preserve the effect of cancellations.
    clean["TotalAmount"] = clean["Quantity"] * clean["UnitPrice"]

    # Temporal variables useful for EDA and later stochastic analysis.
    clean["Date"] = clean["InvoiceDate"].dt.date
    clean["Year"] = clean["InvoiceDate"].dt.year
    clean["Month"] = clean["InvoiceDate"].dt.month
    clean["Day"] = clean["InvoiceDate"].dt.day
    clean["Hour"] = clean["InvoiceDate"].dt.hour
    clean["Weekday"] = clean["InvoiceDate"].dt.day_name()

    return clean


def validate_clean_data(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    """Validate invariants expected from the cleaning pipeline."""
    expected_rows = len(raw) - raw.duplicated().sum()

    assert len(clean) == expected_rows, (
        "Unexpected number of rows after duplicate removal."
    )
    assert clean.duplicated().sum() == 0, (
        "Exact duplicates remain in the clean dataset."
    )
    assert clean["InvoiceNo"].isna().sum() == 0
    assert clean["StockCode"].isna().sum() == 0
    assert clean["InvoiceDate"].isna().sum() == 0
    assert clean["Quantity"].isna().sum() == 0
    assert clean["UnitPrice"].isna().sum() == 0
    assert clean["Country"].isna().sum() == 0

    expected_total = clean["Quantity"] * clean["UnitPrice"]
    assert clean["TotalAmount"].equals(expected_total)

    expected_cancellations = (
        clean["InvoiceNo"].astype(str).str.startswith("C")
    )
    assert clean["IsCancellation"].equals(expected_cancellations)

    print()
    print("Validation: PASSED")


def save_clean_data(df: pd.DataFrame) -> None:
    """Save the cleaned analytical dataset."""
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    print()
    print(f"Saved clean dataset: {PROCESSED_FILE}")

def main() -> None:
    df = load_raw_data()

    print("=== RAW DATASET ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Date range: {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")

    audit = audit_data(df)
    print_audit(audit)

    clean = clean_data(df)
    validate_clean_data(df, clean)
    print()
    print("=== CLEAN DATASET ===")
    print(f"Rows: {len(clean):,}")
    print(f"Removed exact duplicates: {len(df) - len(clean):,}")
    print(f"Columns: {clean.shape[1]}")
    print(f"CustomerID dtype: {clean['CustomerID'].dtype}")
    print(f"TotalAmount min: {clean['TotalAmount'].min():,.2f}")
    print(f"TotalAmount max: {clean['TotalAmount'].max():,.2f}")

    print()
    print("=== TRANSACTION TYPES ===")
    print(clean["TransactionType"].value_counts().to_string())

    save_clean_data(clean)


if __name__ == "__main__":
    main()

