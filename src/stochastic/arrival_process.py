from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "database" / "online_retail.db"


def load_sale_invoices() -> pd.DataFrame:
    """Load one timestamp per sale invoice."""

    query = """
        SELECT
            InvoiceNo,
            MIN(InvoiceDate) AS InvoiceDate
        FROM transactions
        WHERE TransactionType = 'Sale'
        GROUP BY InvoiceNo
    """

    with sqlite3.connect(DATABASE_FILE) as connection:
        df = pd.read_sql_query(
            query,
            connection,
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    return df


def build_daily_counts(
    invoices: pd.DataFrame,
) -> pd.Series:
    """Build calendar-day invoice counts including zero-activity days."""

    counts = (
        invoices
        .assign(
            Date=invoices["InvoiceDate"].dt.normalize()
        )
        .groupby("Date")
        .size()
        .sort_index()
    )

    calendar = pd.date_range(
        counts.index.min(),
        counts.index.max(),
        freq="D",
    )

    counts = (
        counts
        .reindex(
            calendar,
            fill_value=0,
        )
        .astype(int)
    )

    counts.index.name = "Date"

    return counts


def dispersion_statistics(
    counts: pd.Series,
) -> dict:
    """Calculate basic count-process dispersion statistics."""

    mean_count = counts.mean()
    variance = counts.var(
        ddof=1
    )

    return {
        "Days": len(counts),
        "ActiveDays": int(
            (counts > 0).sum()
        ),
        "ZeroDays": int(
            (counts == 0).sum()
        ),
        "Mean": float(mean_count),
        "Variance": float(variance),
        "DispersionIndex": float(
            variance / mean_count
        ),
        "Minimum": int(counts.min()),
        "Median": float(counts.median()),
        "Maximum": int(counts.max()),
    }


def autocorrelations(
    counts: pd.Series,
) -> dict[int, float]:
    """Calculate selected daily autocorrelations."""

    lags = [
        1,
        2,
        7,
        14,
        28,
    ]

    return {
        lag: float(
            counts.autocorr(
                lag=lag
            )
        )
        for lag in lags
    }


def weekday_statistics(
    counts: pd.Series,
) -> pd.DataFrame:
    """Summarize invoice counts by weekday."""

    frame = counts.rename(
        "Invoices"
    ).to_frame()

    frame["WeekdayNumber"] = (
        frame.index.dayofweek
    )

    frame["Weekday"] = (
        frame.index.day_name()
    )

    result = (
        frame
        .groupby(
            [
                "WeekdayNumber",
                "Weekday",
            ]
        )["Invoices"]
        .agg(
            Days="count",
            Mean="mean",
            Median="median",
            Variance="var",
            Minimum="min",
            Maximum="max",
        )
        .reset_index()
        .sort_values(
            "WeekdayNumber"
        )
    )

    return result


def interarrival_statistics(
    invoices: pd.DataFrame,
) -> dict:
    """Calculate time gaps between consecutive sale invoices."""

    timestamps = (
        invoices["InvoiceDate"]
        .sort_values()
        .reset_index(drop=True)
    )

    gaps_seconds = (
        timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    positive_gaps = gaps_seconds[
        gaps_seconds > 0
    ]

    return {
        "Intervals": int(
            len(gaps_seconds)
        ),
        "PositiveIntervals": int(
            len(positive_gaps)
        ),
        "ZeroIntervals": int(
            (gaps_seconds == 0).sum()
        ),
        "MedianMinutes": float(
            positive_gaps.median()
            / 60
        ),
        "MeanMinutes": float(
            positive_gaps.mean()
            / 60
        ),
        "P90Minutes": float(
            positive_gaps.quantile(0.90)
            / 60
        ),
        "MaximumHours": float(
            positive_gaps.max()
            / 3600
        ),
    }


def validate(
    invoices: pd.DataFrame,
    counts: pd.Series,
) -> None:
    """Validate the temporal aggregation."""

    assert not invoices.empty
    assert invoices["InvoiceNo"].is_unique
    assert invoices["InvoiceDate"].notna().all()

    assert not counts.empty
    assert counts.index.is_monotonic_increasing
    assert counts.ge(0).all()

    assert counts.sum() == len(invoices)

    expected_days = (
        counts.index.max()
        - counts.index.min()
    ).days + 1

    assert len(counts) == expected_days


def main() -> None:
    invoices = load_sale_invoices()

    counts = build_daily_counts(
        invoices
    )

    validate(
        invoices,
        counts,
    )

    dispersion = dispersion_statistics(
        counts
    )

    autocorr = autocorrelations(
        counts
    )

    weekday = weekday_statistics(
        counts
    )

    interarrival = interarrival_statistics(
        invoices
    )

    print("=== ARRIVAL PROCESS DATA ===")
    print(
        f"Sale invoices: "
        f"{len(invoices):,}"
    )
    print(
        f"Date range: "
        f"{counts.index.min().date()} "
        f"to "
        f"{counts.index.max().date()}"
    )
    print(
        f"Calendar days: "
        f"{len(counts):,}"
    )
    print()
    print("Temporal validation: PASSED")

    print()
    print("=== DISPERSION ===")

    for key, value in dispersion.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value:,}"
            )

    print()
    print("=== AUTOCORRELATION ===")

    for lag, value in autocorr.items():
        print(
            f"Lag {lag}: {value:.6f}"
        )

    print()
    print("=== WEEKDAY ACTIVITY ===")

    print(
        weekday.to_string(
            index=False,
            formatters={
                "Mean":
                    lambda x: f"{x:.3f}",
                "Median":
                    lambda x: f"{x:.3f}",
                "Variance":
                    lambda x: f"{x:.3f}",
            },
        )
    )

    print()
    print("=== INTERARRIVAL TIMES ===")

    for key, value in interarrival.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value:,}"
            )


if __name__ == "__main__":
    main()
