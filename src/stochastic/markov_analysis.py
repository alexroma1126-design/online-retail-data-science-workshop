from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from arrival_process import (
    PROJECT_ROOT,
    load_sale_invoices,
    build_daily_counts,
)


STATE_ORDER = [
    "Low",
    "Medium",
    "High",
]


def build_activity_states(
    counts: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Create weekday-adjusted activity states."""

    frame = counts.rename(
        "Invoices"
    ).to_frame()

    frame["WeekdayNumber"] = (
        frame.index.dayofweek
    )

    # Saturday is structurally inactive.
    operating = frame[
        frame["WeekdayNumber"] != 5
    ].copy()

    weekday_means = (
        operating
        .groupby("WeekdayNumber")["Invoices"]
        .mean()
    )

    operating["WeekdayMean"] = (
        operating["WeekdayNumber"]
        .map(weekday_means)
    )

    operating["ActivityRatio"] = (
        operating["Invoices"]
        / operating["WeekdayMean"]
    )

    q_low = float(
        operating["ActivityRatio"]
        .quantile(1 / 3)
    )

    q_high = float(
        operating["ActivityRatio"]
        .quantile(2 / 3)
    )

    operating["State"] = pd.cut(
        operating["ActivityRatio"],
        bins=[
            -np.inf,
            q_low,
            q_high,
            np.inf,
        ],
        labels=STATE_ORDER,
        include_lowest=True,
    )

    thresholds = {
        "LowUpper": q_low,
        "MediumUpper": q_high,
    }

    return (
        operating,
        thresholds,
    )


def transition_analysis(
    states: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Estimate first-order Markov transitions."""

    sequence = (
        states[["State"]]
        .copy()
        .sort_index()
    )

    sequence["NextState"] = (
        sequence["State"].shift(-1)
    )

    sequence["NextDate"] = (
        sequence.index.to_series()
        .shift(-1)
    )

    sequence["GapDays"] = (
        sequence["NextDate"]
        - sequence.index.to_series()
    ).dt.days

    # Keep normal operating transitions.
    transitions = sequence[
        sequence["NextState"].notna()
        & sequence["GapDays"].le(3)
    ].copy()

    counts = pd.crosstab(
        transitions["State"],
        transitions["NextState"],
        dropna=False,
    )

    counts = counts.reindex(
        index=STATE_ORDER,
        columns=STATE_ORDER,
        fill_value=0,
    )

    probabilities = (
        counts.div(
            counts.sum(axis=1),
            axis=0,
        )
    )

    total_transitions = int(
        counts.to_numpy().sum()
    )

    diagonal_transitions = int(
        np.trace(
            counts.to_numpy()
        )
    )

    persistence = (
        diagonal_transitions
        / total_transitions
    )

    chi2, p_value, dof, _ = (
        chi2_contingency(
            counts.to_numpy()
        )
    )

    matrix = probabilities.to_numpy(
        dtype=float
    )

    eigenvalues, eigenvectors = (
        np.linalg.eig(
            matrix.T
        )
    )

    index = int(
        np.argmin(
            np.abs(
                eigenvalues - 1
            )
        )
    )

    stationary = np.real(
        eigenvectors[:, index]
    )

    if stationary.sum() < 0:
        stationary *= -1

    stationary = (
        stationary
        / stationary.sum()
    )

    diagnostics = {
        "Transitions":
            total_transitions,
        "Persistence":
            float(persistence),
        "ChiSquare":
            float(chi2),
        "DegreesFreedom":
            int(dof),
        "ChiSquarePValue":
            float(p_value),
        "StationaryLow":
            float(stationary[0]),
        "StationaryMedium":
            float(stationary[1]),
        "StationaryHigh":
            float(stationary[2]),
    }

    return (
        counts,
        probabilities,
        diagnostics,
    )


def state_summary(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize empirical activity states."""

    summary = (
        states
        .groupby(
            "State",
            observed=False,
        )
        .agg(
            Days=("Invoices", "size"),
            MeanInvoices=("Invoices", "mean"),
            MedianInvoices=("Invoices", "median"),
            MeanActivityRatio=(
                "ActivityRatio",
                "mean",
            ),
        )
        .reindex(
            STATE_ORDER
        )
        .reset_index()
    )

    return summary


def save_outputs(
    transition_probabilities: pd.DataFrame,
    state_table: pd.DataFrame,
    thresholds: dict,
    diagnostics: dict,
) -> tuple[Path, Path]:
    """Save Markov heatmap and Spanish report."""

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
        / "07_markov_transition_matrix.png"
    )

    matrix = (
        transition_probabilities
        .to_numpy(
            dtype=float
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(
        range(3),
        labels=[
            "Baja",
            "Media",
            "Alta",
        ],
    )

    ax.set_yticks(
        range(3),
        labels=[
            "Baja",
            "Media",
            "Alta",
        ],
    )

    ax.set_xlabel(
        "Estado siguiente"
    )

    ax.set_ylabel(
        "Estado actual"
    )

    ax.set_title(
        "Matriz de transici\u00f3n "
        "de actividad diaria"
    )

    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Probabilidad",
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    report_path = (
        reports_dir
        / "markov_activity_report.txt"
    )

    lines = [
        "AN\u00c1LISIS DE MARKOV DE LA ACTIVIDAD DIARIA",
        "=" * 49,
        "",
        "1. Definici\u00f3n de estados",
        "",
        (
            "Se excluye el s\u00e1bado porque presenta "
            "actividad estructuralmente nula."
        ),
        (
            "Cada d\u00eda operativo se normaliza respecto "
            "a la media hist\u00f3rica de su mismo d\u00eda "
            "de la semana."
        ),
        (
            "Los estados Bajo, Medio y Alto se definen "
            "mediante terciles de esta actividad relativa."
        ),
        "",
        (
            f"L\u00edmite Bajo/Medio: "
            f"{thresholds['LowUpper']:.6f}"
        ),
        (
            f"L\u00edmite Medio/Alto: "
            f"{thresholds['MediumUpper']:.6f}"
        ),
        "",
        "2. Resumen de estados",
        "",
        state_table.to_string(
            index=False,
            formatters={
                "MeanInvoices":
                    lambda x: f"{x:.3f}",
                "MedianInvoices":
                    lambda x: f"{x:.3f}",
                "MeanActivityRatio":
                    lambda x: f"{x:.3f}",
            },
        ),
        "",
        "3. Diagn\u00f3sticos",
        "",
    ]

    for key, value in diagnostics.items():
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
            "4. Interpretaci\u00f3n",
            "",
            (
                "La matriz de transici\u00f3n permite "
                "evaluar si el estado de actividad actual "
                "contiene informaci\u00f3n sobre el siguiente."
            ),
            (
                "Una probabilidad diagonal elevada indica "
                "persistencia del mismo r\u00e9gimen."
            ),
            (
                "La prueba chi-cuadrado contrasta la "
                "independencia entre el estado actual y "
                "el estado siguiente."
            ),
            (
                "Este modelo es descriptivo y no implica "
                "que el proceso real satisfaga exactamente "
                "la propiedad de Markov."
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

    states, thresholds = (
        build_activity_states(
            counts
        )
    )

    state_table = (
        state_summary(
            states
        )
    )

    transition_counts, transition_probabilities, diagnostics = (
        transition_analysis(
            states
        )
    )

    figure_path, report_path = (
        save_outputs(
            transition_probabilities,
            state_table,
            thresholds,
            diagnostics,
        )
    )

    print("=== MARKOV STATES ===")

    print(
        f"Low/Medium threshold: "
        f"{thresholds['LowUpper']:.6f}"
    )

    print(
        f"Medium/High threshold: "
        f"{thresholds['MediumUpper']:.6f}"
    )

    print()

    print(
        state_table.to_string(
            index=False,
            formatters={
                "MeanInvoices":
                    lambda x: f"{x:.3f}",
                "MedianInvoices":
                    lambda x: f"{x:.3f}",
                "MeanActivityRatio":
                    lambda x: f"{x:.3f}",
            },
        )
    )

    print()
    print("=== TRANSITION COUNTS ===")
    print(
        transition_counts.to_string()
    )

    print()
    print(
        "=== TRANSITION PROBABILITIES ==="
    )

    print(
        transition_probabilities.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=== MARKOV DIAGNOSTICS ===")

    for key, value in diagnostics.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.8f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print()
    print("=== OUTPUT FILES ===")
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
