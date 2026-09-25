from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from arrival_process import (
    PROJECT_ROOT,
    load_sale_invoices,
    build_daily_counts,
)


def weekday_poisson_diagnostics(
    counts: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate Poisson dispersion after weekday adjustment."""

    frame = counts.rename(
        "Invoices"
    ).to_frame()

    frame["WeekdayNumber"] = (
        frame.index.dayofweek
    )

    weekday_stats = (
        frame
        .groupby("WeekdayNumber")["Invoices"]
        .agg(
            Days="count",
            Mean="mean",
            Variance="var",
        )
    )

    weekday_stats["Dispersion"] = (
        weekday_stats["Variance"]
        / weekday_stats["Mean"]
    )

    weekday_stats.loc[
        weekday_stats["Mean"] == 0,
        "Dispersion",
    ] = np.nan

    operating = frame[
        frame["WeekdayNumber"] != 5
    ].copy()

    weekday_means = (
        operating
        .groupby("WeekdayNumber")["Invoices"]
        .mean()
    )

    operating["Expected"] = (
        operating["WeekdayNumber"]
        .map(weekday_means)
    )

    pearson_terms = (
        (
            operating["Invoices"]
            - operating["Expected"]
        ) ** 2
        / operating["Expected"]
    )

    pearson_x2 = float(
        pearson_terms.sum()
    )

    residual_df = (
        len(operating)
        - len(weekday_means)
    )

    pearson_dispersion = (
        pearson_x2
        / residual_df
    )

    p_value = float(
        chi2.sf(
            pearson_x2,
            residual_df,
        )
    )

    observed_zero_days = int(
        (operating["Invoices"] == 0).sum()
    )

    expected_zero_days = float(
        np.exp(
            -operating["Expected"]
        ).sum()
    )

    diagnostics = {
        "OperatingCalendarDays":
            len(operating),
        "ObservedZeroDays":
            observed_zero_days,
        "ExpectedZeroDaysPoisson":
            expected_zero_days,
        "PearsonChiSquare":
            pearson_x2,
        "ResidualDegreesFreedom":
            residual_df,
        "PearsonDispersion":
            pearson_dispersion,
        "ChiSquarePValue":
            p_value,
    }

    return (
        weekday_stats.reset_index(),
        diagnostics,
    )


def within_day_interarrivals(
    invoices: pd.DataFrame,
) -> dict:
    """Analyze invoice gaps within the same calendar day."""

    timestamps = (
        invoices["InvoiceDate"]
        .sort_values()
        .reset_index(drop=True)
    )

    previous = timestamps.shift(1)

    same_day = (
        timestamps.dt.normalize()
        == previous.dt.normalize()
    )

    gaps_minutes = (
        timestamps
        - previous
    ).dt.total_seconds() / 60

    gaps = gaps_minutes[
        same_day
        & gaps_minutes.notna()
    ]

    zero_gaps = gaps[
        gaps == 0
    ]

    positive = gaps[
        gaps > 0
    ]

    mean_gap = float(
        positive.mean()
    )

    std_gap = float(
        positive.std(ddof=1)
    )

    return {
        "WithinDayIntervals":
            int(len(gaps)),
        "ZeroMinuteIntervals":
            int(len(zero_gaps)),
        "PositiveIntervals":
            int(len(positive)),
        "MeanMinutes":
            mean_gap,
        "MedianMinutes":
            float(positive.median()),
        "P90Minutes":
            float(
                positive.quantile(0.90)
            ),
        "CoefficientVariation":
            std_gap / mean_gap,
    }


def save_outputs(
    counts: pd.Series,
    weekday_stats: pd.DataFrame,
    poisson: dict,
    interarrival: dict,
) -> tuple[Path, Path]:

    figures_dir = (
        PROJECT_ROOT / "figures"
    )

    reports_dir = (
        PROJECT_ROOT / "reports"
    )

    figures_dir.mkdir(
        exist_ok=True,
    )

    reports_dir.mkdir(
        exist_ok=True,
    )

    figure_path = (
        figures_dir
        / "06_arrival_diagnostics.png"
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        counts.index,
        counts.values,
        linewidth=1,
    )

    plt.xlabel("Fecha")
    plt.ylabel(
        "Facturas de venta por d\u00eda"
    )

    plt.title(
        "Proceso diario de llegada de facturas"
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    report_path = (
        reports_dir
        / "stochastic_arrivals_report.txt"
    )

    lines = [
        "AN\u00c1LISIS ESTOC\u00c1STICO DE LLEGADAS",
        "=" * 44,
        "",
        "1. Objetivo",
        "",
        (
            "Se estudia si la llegada diaria de facturas "
            "puede aproximarse mediante un proceso de "
            "Poisson homog\u00e9neo."
        ),
        "",
        "2. Patr\u00f3n semanal",
        "",
        weekday_stats.to_string(
            index=False,
            formatters={
                "Mean":
                    lambda x: f"{x:.3f}",
                "Variance":
                    lambda x: f"{x:.3f}",
                "Dispersion":
                    lambda x: (
                        "NA"
                        if pd.isna(x)
                        else f"{x:.3f}"
                    ),
            },
        ),
        "",
        (
            "El s\u00e1bado presenta actividad "
            "estructuralmente nula y por ello se excluye "
            "de la comparaci\u00f3n Poisson ajustada."
        ),
        "",
        "3. Diagn\u00f3stico Poisson ajustado por d\u00eda de semana",
        "",
    ]

    for key, value in poisson.items():
        if isinstance(value, float):
            lines.append(
                f"{key}: {value:.8f}"
            )
        else:
            lines.append(
                f"{key}: {value}"
            )

    lines.extend(
        [
            "",
            "4. Intervalos de llegada dentro del mismo d\u00eda",
            "",
        ]
    )

    for key, value in interarrival.items():
        if isinstance(value, float):
            lines.append(
                f"{key}: {value:.6f}"
            )
        else:
            lines.append(
                f"{key}: {value}"
            )

    lines.extend(
        [
            "",
            "5. Interpretaci\u00f3n",
            "",
            (
                "En un modelo Poisson homog\u00e9neo ideal, "
                "la media y la varianza de los conteos "
                "deber\u00edan ser aproximadamente iguales."
            ),
            (
                "Los datos presentan una varianza muy "
                "superior a la media, incluso al separar "
                "la actividad por d\u00eda de la semana."
            ),
            (
                "La fuerte autocorrelaci\u00f3n semanal "
                "observada previamente y las diferencias "
                "entre d\u00edas muestran que la intensidad "
                "de llegada no es constante."
            ),
            (
                "Por tanto, un proceso de Poisson "
                "homog\u00e9neo resulta demasiado simple "
                "para representar estas llegadas."
            ),
            (
                "La evidencia es compatible con una "
                "intensidad variable en el tiempo y con "
                "sobredispersi\u00f3n adicional."
            ),
        ]
    )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return (
        figure_path,
        report_path,
    )


def main() -> None:
    invoices = load_sale_invoices()

    counts = build_daily_counts(
        invoices
    )

    weekday_stats, poisson = (
        weekday_poisson_diagnostics(
            counts
        )
    )

    interarrival = (
        within_day_interarrivals(
            invoices
        )
    )

    figure_path, report_path = (
        save_outputs(
            counts,
            weekday_stats,
            poisson,
            interarrival,
        )
    )

    print(
        "=== WEEKDAY POISSON DIAGNOSTICS ==="
    )

    print(
        weekday_stats.to_string(
            index=False,
            formatters={
                "Mean":
                    lambda x: f"{x:.3f}",
                "Variance":
                    lambda x: f"{x:.3f}",
                "Dispersion":
                    lambda x: (
                        "NA"
                        if pd.isna(x)
                        else f"{x:.3f}"
                    ),
            },
        )
    )

    print()
    print(
        "=== WEEKDAY-ADJUSTED POISSON ==="
    )

    for key, value in poisson.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.8f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print()
    print(
        "=== WITHIN-DAY INTERARRIVALS ==="
    )

    for key, value in interarrival.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print()
    print("=== OUTPUT FILES ===")
    print(
        f"Figure: {figure_path}"
    )
    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
