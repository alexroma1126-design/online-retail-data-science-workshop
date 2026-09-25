"""Create the SQLite database from the cleaned Online Retail dataset."""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "online_retail_clean.csv"
)
DATABASE_FILE = PROJECT_ROOT / "database" / "online_retail.db"

TABLE_NAME = "transactions"


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned analytical dataset."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {PROCESSED_FILE}"
        )

    df = pd.read_csv(
        PROCESSED_FILE,
        parse_dates=["InvoiceDate", "Date"],
    )

    df["CustomerID"] = df["CustomerID"].astype("Int64")

    return df


def create_database(df: pd.DataFrame) -> None:
    """Create SQLite database and store the transaction table."""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_FILE) as connection:
        df.to_sql(
            TABLE_NAME,
            connection,
            if_exists="replace",
            index=False,
        )


def create_indexes() -> None:
    """Create indexes for fields frequently used in later analysis."""
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_invoice_no ON transactions(InvoiceNo)",
        "CREATE INDEX IF NOT EXISTS idx_invoice_date ON transactions(InvoiceDate)",
        "CREATE INDEX IF NOT EXISTS idx_customer_id ON transactions(CustomerID)",
        "CREATE INDEX IF NOT EXISTS idx_country ON transactions(Country)",
        "CREATE INDEX IF NOT EXISTS idx_transaction_type ON transactions(TransactionType)",
    ]

    with sqlite3.connect(DATABASE_FILE) as connection:
        for statement in statements:
            connection.execute(statement)

        connection.commit()


def validate_database(expected_rows: int) -> None:
    """Validate row counts and transaction categories stored in SQLite."""
    with sqlite3.connect(DATABASE_FILE) as connection:
        stored_rows = connection.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME}"
        ).fetchone()[0]

        transaction_counts = dict(
            connection.execute(
                f"""
                SELECT TransactionType, COUNT(*)
                FROM {TABLE_NAME}
                GROUP BY TransactionType
                """
            ).fetchall()
        )

        indexes = connection.execute(
            f"PRAGMA index_list({TABLE_NAME})"
        ).fetchall()

    assert stored_rows == expected_rows, (
        f"Row-count mismatch: expected {expected_rows}, found {stored_rows}."
    )

    assert sum(transaction_counts.values()) == expected_rows
    assert set(transaction_counts) == {
        "Sale",
        "Cancellation",
        "Adjustment",
        "Other",
    }

    print()
    print("=== DATABASE VALIDATION ===")
    print(f"Rows stored: {stored_rows:,}")

    for category, count in sorted(
        transaction_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{category:15s}: {count:,}")

    print(f"Indexes created: {len(indexes)}")
    print("Database validation: PASSED")

def main() -> None:
    df = load_clean_data()

    print("=== SQLITE DATABASE CREATION ===")
    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {df.shape[1]}")

    create_database(df)
    create_indexes()
    validate_database(len(df))

    print(f"Database created: {DATABASE_FILE}")
    print(f"Table created: {TABLE_NAME}")


if __name__ == "__main__":
    main()

