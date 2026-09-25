from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
)

from statsmodels.tsa.statespace.sarimax import (
    SARIMAX,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from arrival_process import (
    PROJECT_ROOT,
    load_sale_invoices,
    build_daily_counts,
)


TEST_DAYS = 56
SEASONAL_PERIOD = 7

CANDIDATES = [
    (
        (1, 0, 0),
        (1, 1, 0, 7),
    ),
    (
        (1, 0, 1),
        (1, 1, 0, 7),
    ),
    (
        (1, 0, 0),
        (0, 1, 1, 7),
    ),
    (
        (1, 0, 1),
        (0, 1, 1, 7),
    ),
    (
        (2, 0, 1),
        (1, 1, 0, 7),
    ),
    (
        (1, 0, 1),
        (1, 1, 1, 7),
    ),
]


def split_series(
    counts: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Chronological train-test split."""

    if len(counts) <= TEST_DAYS + SEASONAL_PERIOD:
        raise ValueError(
            "Time series is too short."
        )

    train = counts.iloc[
        :-TEST_DAYS
    ].astype(float)

    test = counts.iloc[
        -TEST_DAYS:
    ].astype(float)

    assert len(test) == TEST_DAYS

    assert (
        train.index.max()
        < test.index.min()
    )

    return (
        train,
        test,
    )


def seasonal_naive_forecast(
    train: pd.Series,
    test: pd.Series,
) -> pd.Series:
    """Repeat the final observed training week."""

    last_week = (
        train.iloc[
            -SEASONAL_PERIOD:
        ]
        .to_numpy(
            dtype=float
        )
    )

    repeated = np.resize(
        last_week,
        len(test),
    )

    return pd.Series(
        repeated,
        index=test.index,
        name="SeasonalNaive",
    )


def fit_sarima_candidates(
    train: pd.Series,
) -> tuple[
    pd.DataFrame,
    object,
    tuple,
    tuple,
]:
    """Fit a compact SARIMA candidate set using training data only."""

    transformed = np.log1p(
        train
    )

    rows = []
    successful_models = []

    for order, seasonal_order in CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                model = SARIMAX(
                    transformed,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                result = model.fit(
                    disp=False,
                    maxiter=300,
                )

            rows.append(
                {
                    "Order": str(order),
                    "SeasonalOrder":
                        str(seasonal_order),
                    "AIC": float(
                        result.aic
                    ),
                    "BIC": float(
                        result.bic
                    ),
                    "Converged": bool(
                        result.mle_retvals.get(
                            "converged",
                            False,
                        )
                    ),
                }
            )

            successful_models.append(
                (
                    float(result.aic),
                    result,
                    order,
                    seasonal_order,
                )
            )

        except Exception as exc:
            rows.append(
                {
                    "Order": str(order),
                    "SeasonalOrder":
                        str(seasonal_order),
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Converged": False,
                    "Error": str(exc),
                }
            )

    if not successful_models:
        raise RuntimeError(
            "No SARIMA candidate converged."
        )

    successful_models.sort(
        key=lambda item: item[0]
    )

    _, best_result, best_order, best_seasonal = (
        successful_models[0]
    )

    table = (
        pd.DataFrame(rows)
        .sort_values(
            "AIC",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return (
        table,
        best_result,
        best_order,
        best_seasonal,
    )


def sarima_forecast(
    result,
    test: pd.Series,
) -> pd.Series:
    """Forecast on original count scale."""

    forecast_log = (
        result
        .get_forecast(
            steps=len(test)
        )
        .predicted_mean
    )

    forecast = np.expm1(
        forecast_log
    )

    forecast = np.clip(
        forecast,
        0,
        None,
    )

    forecast_series = pd.Series(
        np.asarray(
            forecast,
            dtype=float,
        ),
        index=test.index,
        name="SARIMA",
    )

    # Saturdays are structurally inactive in the
    # complete observed history.
    forecast_series.loc[
        forecast_series.index.dayofweek == 5
    ] = 0.0

    return forecast_series


def smape(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Symmetric mean absolute percentage error."""

    actual_values = actual.to_numpy(
        dtype=float
    )

    forecast_values = forecast.to_numpy(
        dtype=float
    )

    denominator = (
        np.abs(actual_values)
        + np.abs(forecast_values)
    )

    numerator = (
        2
        * np.abs(
            actual_values
            - forecast_values
        )
    )

    valid = denominator > 0

    return float(
        np.mean(
            numerator[valid]
            / denominator[valid]
        )
        * 100
    )


def metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict:
    """Calculate out-of-sample forecast metrics."""

    mae = mean_absolute_error(
        actual,
        forecast,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            forecast,
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE": smape(
            actual,
            forecast,
        ),
    }


def residual_diagnostics(
    result,
) -> pd.DataFrame:
    """Ljung-Box diagnostics for fitted SARIMA residuals."""

    residuals = pd.Series(
        result.resid
    ).dropna()

    residuals = residuals.iloc[
        14:
    ]

    return acorr_ljungbox(
        residuals,
        lags=[
            7,
            14,
        ],
        return_df=True,
    )


def save_outputs(
    train: pd.Series,
    test: pd.Series,
    baseline: pd.Series,
    sarima: pd.Series,
    candidate_table: pd.DataFrame,
    baseline_metrics: dict,
    sarima_metrics: dict,
    best_order: tuple,
    best_seasonal: tuple,
    ljung_box: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save forecast figure and Spanish report."""

    figures_dir = (
        PROJECT_ROOT
        / "figures"
    )

    reports_dir = (
        PROJECT_ROOT
        / "reports"
    )

    figures_dir.mkdir(
        exist_ok=True,
    )

    reports_dir.mkdir(
        exist_ok=True,
    )

    figure_path = (
        figures_dir
        / "08_sarima_forecast.png"
    )

    context = train.iloc[
        -42:
    ]

    plt.figure(
        figsize=(13, 6)
    )

    plt.plot(
        context.index,
        context.values,
        label="Entrenamiento reciente",
        linewidth=1.2,
    )

    plt.plot(
        test.index,
        test.values,
        label="Real",
        linewidth=1.6,
    )

    plt.plot(
        baseline.index,
        baseline.values,
        label="Baseline semanal",
        linestyle="--",
    )

    plt.plot(
        sarima.index,
        sarima.values,
        label="SARIMA",
        linestyle="--",
    )

    plt.xlabel(
        "Fecha"
    )

    plt.ylabel(
        "Facturas de venta por d\u00eda"
    )

    plt.title(
        "Pron\u00f3stico fuera de muestra: "
        "baseline semanal vs SARIMA"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    mae_improvement = (
        100
        * (
            baseline_metrics["MAE"]
            - sarima_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
    )

    rmse_improvement = (
        100
        * (
            baseline_metrics["RMSE"]
            - sarima_metrics["RMSE"]
        )
        / baseline_metrics["RMSE"]
    )

    report_path = (
        reports_dir
        / "time_series_report.txt"
    )

    lines = [
        "AN\u00c1LISIS DE SERIES TEMPORALES Y SARIMA",
        "=" * 46,
        "",
        "1. Dise\u00f1o de validaci\u00f3n",
        "",
        (
            f"Observaciones de entrenamiento: "
            f"{len(train)}"
        ),
        (
            f"Observaciones de prueba: "
            f"{len(test)}"
        ),
        (
            f"Inicio de prueba: "
            f"{test.index.min().date()}"
        ),
        (
            f"Fin de prueba: "
            f"{test.index.max().date()}"
        ),
        (
            "La prueba corresponde a las \u00faltimas "
            "ocho semanas calendario."
        ),
        (
            "El conjunto de prueba no se utiliza para "
            "seleccionar el modelo SARIMA."
        ),
        "",
        "2. Modelos SARIMA candidatos",
        "",
        candidate_table.to_string(
            index=False,
            formatters={
                "AIC":
                    lambda x: (
                        "NA"
                        if pd.isna(x)
                        else f"{x:.3f}"
                    ),
                "BIC":
                    lambda x: (
                        "NA"
                        if pd.isna(x)
                        else f"{x:.3f}"
                    ),
            },
        ),
        "",
        (
            f"Modelo seleccionado: "
            f"SARIMA{best_order}"
            f"{best_seasonal}"
        ),
        "",
        "3. Evaluaci\u00f3n fuera de muestra",
        "",
        (
            "Baseline semanal: se repite el "
            "\u00faltimo patr\u00f3n observado de siete d\u00edas."
        ),
        "",
        (
            f"Baseline MAE: "
            f"{baseline_metrics['MAE']:.6f}"
        ),
        (
            f"Baseline RMSE: "
            f"{baseline_metrics['RMSE']:.6f}"
        ),
        (
            f"Baseline sMAPE: "
            f"{baseline_metrics['sMAPE']:.3f}%"
        ),
        "",
        (
            f"SARIMA MAE: "
            f"{sarima_metrics['MAE']:.6f}"
        ),
        (
            f"SARIMA RMSE: "
            f"{sarima_metrics['RMSE']:.6f}"
        ),
        (
            f"SARIMA sMAPE: "
            f"{sarima_metrics['sMAPE']:.3f}%"
        ),
        "",
        (
            f"Mejora MAE de SARIMA frente al baseline: "
            f"{mae_improvement:.3f}%"
        ),
        (
            f"Mejora RMSE de SARIMA frente al baseline: "
            f"{rmse_improvement:.3f}%"
        ),
        "",
        "4. Diagn\u00f3stico de residuos",
        "",
        ljung_box.to_string(
            float_format=lambda x: f"{x:.6f}"
        ),
        "",
        "5. Interpretaci\u00f3n",
        "",
        (
            "La comparaci\u00f3n fuera de muestra permite "
            "evaluar si el modelo temporal aporta "
            "informaci\u00f3n adicional frente a repetir "
            "simplemente el patr\u00f3n de la semana anterior."
        ),
        (
            "Un valor positivo en las mejoras de MAE o "
            "RMSE indica ventaja de SARIMA; un valor "
            "negativo indica que el baseline semanal fue "
            "m\u00e1s preciso."
        ),
        (
            "La conclusi\u00f3n se basa en datos no usados "
            "durante la estimaci\u00f3n del modelo."
        ),
    ]

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

    train, test = split_series(
        counts
    )

    baseline = seasonal_naive_forecast(
        train,
        test,
    )

    (
        candidate_table,
        best_result,
        best_order,
        best_seasonal,
    ) = fit_sarima_candidates(
        train
    )

    sarima = sarima_forecast(
        best_result,
        test,
    )

    baseline_metrics = metrics(
        test,
        baseline,
    )

    sarima_metrics = metrics(
        test,
        sarima,
    )

    ljung_box = residual_diagnostics(
        best_result
    )

    figure_path, report_path = (
        save_outputs(
            train,
            test,
            baseline,
            sarima,
            candidate_table,
            baseline_metrics,
            sarima_metrics,
            best_order,
            best_seasonal,
            ljung_box,
        )
    )

    print(
        "=== TIME SERIES SPLIT ==="
    )

    print(
        f"Train observations: "
        f"{len(train)}"
    )

    print(
        f"Test observations: "
        f"{len(test)}"
    )

    print(
        f"Train end: "
        f"{train.index.max().date()}"
    )

    print(
        f"Test start: "
        f"{test.index.min().date()}"
    )

    print(
        f"Test end: "
        f"{test.index.max().date()}"
    )

    print()
    print(
        "=== SARIMA CANDIDATES ==="
    )

    print(
        candidate_table.to_string(
            index=False,
            formatters={
                "AIC":
                    lambda x: (
                        "NA"
                        if pd.isna(x)
                        else f"{x:.3f}"
                    ),
                "BIC":
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
        "=== SELECTED SARIMA ==="
    )

    print(
        f"Order: {best_order}"
    )

    print(
        f"Seasonal order: "
        f"{best_seasonal}"
    )

    print()
    print(
        "=== OUT-OF-SAMPLE METRICS ==="
    )

    print(
        f"Baseline MAE: "
        f"{baseline_metrics['MAE']:.6f}"
    )

    print(
        f"Baseline RMSE: "
        f"{baseline_metrics['RMSE']:.6f}"
    )

    print(
        f"Baseline sMAPE: "
        f"{baseline_metrics['sMAPE']:.3f}%"
    )

    print(
        f"SARIMA MAE: "
        f"{sarima_metrics['MAE']:.6f}"
    )

    print(
        f"SARIMA RMSE: "
        f"{sarima_metrics['RMSE']:.6f}"
    )

    print(
        f"SARIMA sMAPE: "
        f"{sarima_metrics['sMAPE']:.3f}%"
    )

    mae_improvement = (
        100
        * (
            baseline_metrics["MAE"]
            - sarima_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
    )

    rmse_improvement = (
        100
        * (
            baseline_metrics["RMSE"]
            - sarima_metrics["RMSE"]
        )
        / baseline_metrics["RMSE"]
    )

    print(
        f"MAE improvement: "
        f"{mae_improvement:.3f}%"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.3f}%"
    )

    print()
    print(
        "=== LJUNG-BOX RESIDUAL TEST ==="
    )

    print(
        ljung_box.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print(
        "=== OUTPUT FILES ==="
    )

    print(
        f"Figure: {figure_path}"
    )

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
