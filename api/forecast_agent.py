"""
forecast_agent.py — OnePilot Forecasting Module
Sprint 14 — Intégration SARIMA + Prophet + LSTM + Ensemble
"""
import asyncio
import logging
import time
from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
LSTM_BEST_PARAMS = {
    "window": 8,
    "hidden": 128,
    "epochs": 150,
    "lr":     0.0005,
    "batch":  16,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SQL templates par type d'entité
SQL_TEMPLATES = {
    "ventes_weekly": """
        SELECT
            YEAR(o.OrderDate)           AS annee,
            DATEPART(WEEK, o.OrderDate) AS semaine,
            MIN(o.OrderDate)            AS date_debut,
            ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS valeur
        FROM Orders o
        JOIN [Order Details] od ON o.OrderID = od.OrderID
        WHERE o.OrderDate IS NOT NULL
        GROUP BY YEAR(o.OrderDate), DATEPART(WEEK, o.OrderDate)
        ORDER BY annee, semaine
    """,
    "ventes_monthly": """
        SELECT
            CAST(YEAR(o.OrderDate) AS VARCHAR) + '-' +
                RIGHT('0' + CAST(MONTH(o.OrderDate) AS VARCHAR), 2) AS date_debut,
            ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS valeur
        FROM Orders o
        JOIN [Order Details] od ON o.OrderID = od.OrderID
        WHERE o.OrderDate IS NOT NULL
        GROUP BY YEAR(o.OrderDate), MONTH(o.OrderDate)
        ORDER BY YEAR(o.OrderDate), MONTH(o.OrderDate)
    """,
    "tresorerie_weekly": """
        SELECT
            YEAR(TRNDATE)           AS annee,
            DATEPART(WEEK, TRNDATE) AS semaine,
            MIN(TRNDATE)            AS date_debut,
            SUM([montant avec signe]) AS valeur
        FROM [Transactions bancaires]
        GROUP BY YEAR(TRNDATE), DATEPART(WEEK, TRNDATE)
        ORDER BY annee, semaine
    """,
    "tresorerie_monthly": """
        SELECT
            CAST(YEAR(TRNDATE) AS VARCHAR) + '-' +
                RIGHT('0' + CAST(MONTH(TRNDATE) AS VARCHAR), 2) AS date_debut,
            SUM([montant avec signe]) AS valeur
        FROM [Transactions bancaires]
        GROUP BY YEAR(TRNDATE), MONTH(TRNDATE)
        ORDER BY YEAR(TRNDATE), MONTH(TRNDATE)
    """,
}

# ──────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE LSTM
# ──────────────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            1, hidden, layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTION DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────
def detect_entity_and_sql(source_name: str, question: str, granularity: str) -> str:
    """Détecte l'entité à forecaster et retourne le SQL adapté."""
    q = question.lower()
    is_tresorerie = any(w in q for w in [
        "trésor", "tresor", "bancaire", "solde", "flux", "liquidit",
        "encaissement", "décaissement"
    ])
    key = f"{'tresorerie' if is_tresorerie else 'ventes'}_{granularity}"
    return SQL_TEMPLATES.get(key, SQL_TEMPLATES[f"ventes_{granularity}"])


def extract_time_series(rows: list, granularity: str) -> pd.Series:
    """Convertit les rows API en pd.Series temporelle."""
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Aucune donnée retournée par la requête SQL")

    df["date"]  = pd.to_datetime(df["date_debut"])
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce").fillna(0)
    df = df.set_index("date").sort_index()
    series = df["valeur"]

    if len(series) < 20:
        raise ValueError(f"Série trop courte ({len(series)} points) pour le forecasting")

    return series


# ──────────────────────────────────────────────────────────────────────────────
# MODÈLES
# ──────────────────────────────────────────────────────────────────────────────
def run_sarima(train: pd.Series, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """SARIMA via auto_arima + statsmodels pour la prévision."""
    try:
        from pmdarima import auto_arima
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

        model = auto_arima(
            train, seasonal=False, stepwise=True,
            max_p=3, max_q=3, d=None,
            information_criterion="aic",
            suppress_warnings=True, error_action="ignore",
        )
        sm = SM_ARIMA(train.values, order=model.order).fit()
        fc_result = sm.get_forecast(n_total)
        fc   = fc_result.predicted_mean
        ci   = fc_result.conf_int(alpha=0.05)
        fc   = np.where(np.isnan(fc), train.mean(), fc)
        std  = train.std()
        ci[np.isnan(ci[:, 0]), 0] = fc[np.isnan(ci[:, 0])] - std
        ci[np.isnan(ci[:, 1]), 1] = fc[np.isnan(ci[:, 1])] + std
        return fc, ci

    except Exception as e:
        logger.warning(f"[SARIMA] Erreur : {e} — fallback moyenne mobile")
        fc = np.full(n_total, train.rolling(4, min_periods=1).mean().iloc[-1])
        std = train.std()
        return fc, np.column_stack([fc - std, fc + std])


def run_prophet(train: pd.Series, n_total: int, freq: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prophet avec paramètres adaptés à la longueur de la série."""
    try:
        import logging as _log
        _log.getLogger("prophet").setLevel(_log.ERROR)
        _log.getLogger("cmdstanpy").setLevel(_log.ERROR)
        from prophet import Prophet

        df_p = pd.DataFrame({"ds": train.index, "y": train.values})
        m = Prophet(
            yearly_seasonality=(len(train) >= 52),
            weekly_seasonality=(freq == "W"),
            daily_seasonality=False,
            changepoint_prior_scale=0.3,
            interval_width=0.95,
        )
        m.fit(df_p)
        future = m.make_future_dataframe(periods=n_total, freq=freq)
        fc     = m.predict(future)
        fut    = fc[fc["ds"] > train.index[-1]].head(n_total)
        return fut["yhat"].values, fut["yhat_lower"].values, fut["yhat_upper"].values

    except Exception as e:
        logger.warning(f"[Prophet] Erreur : {e} — fallback")
        fc = np.full(n_total, train.mean())
        std = train.std()
        return fc, fc - std, fc + std


def run_lstm(
    train: pd.Series,
    n_total: int,
    window: int   = LSTM_BEST_PARAMS["window"],
    hidden: int   = LSTM_BEST_PARAMS["hidden"],
    epochs: int   = LSTM_BEST_PARAMS["epochs"],
    lr: float     = LSTM_BEST_PARAMS["lr"],
    batch: int    = LSTM_BEST_PARAMS["batch"],
) -> np.ndarray:
    """LSTM Vanilla 2 couches avec les paramètres fine-tunés."""
    try:
        has_negatives = (train.min() < 0)
        feature_range = (-1, 1) if has_negatives else (0, 1)
        scaler   = MinMaxScaler(feature_range=feature_range)
        train_sc = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

        X_lst, y_lst = [], []
        for i in range(len(train_sc) - window):
            X_lst.append(train_sc[i:i + window])
            y_lst.append(train_sc[i + window])

        X_t = torch.FloatTensor(np.array(X_lst)).unsqueeze(-1).to(DEVICE)
        y_t = torch.FloatTensor(np.array(y_lst)).to(DEVICE)

        model  = LSTMForecaster(hidden=hidden).to(DEVICE)
        opt    = torch.optim.Adam(model.parameters(), lr=lr)
        crit   = nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=batch, shuffle=True,
        )
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

        # Prévision récursive
        model.eval()
        win   = list(train_sc[-window:])
        preds = []
        lo, hi = feature_range
        with torch.no_grad():
            for _ in range(n_total):
                x_in = torch.FloatTensor(win[-window:]).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                pred = model(x_in).item()
                preds.append(pred)
                win.append(pred)

        fc = scaler.inverse_transform(
            np.clip(preds, lo, hi).reshape(-1, 1)
        ).flatten()
        return fc

    except Exception as e:
        logger.warning(f"[LSTM] Erreur : {e} — fallback")
        return np.full(n_total, train.mean())


# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE 7 COMBINAISONS
# ──────────────────────────────────────────────────────────────────────────────
def run_ensemble(
    train: pd.Series,
    test: pd.Series,
    n_horizon: int,
    freq: str,
) -> dict:
    """
    Entraîne les 3 modèles, calcule les 7 combinaisons,
    sélectionne par MAE minimum et retourne les prévisions finales.
    """
    n_test  = len(test)
    n_total = n_test + n_horizon
    y       = test.values

    logger.info(f"[Ensemble] Train={len(train)} Test={n_test} Horizon={n_horizon}")
    t0 = time.time()

    # 1. SARIMA
    sarima_fc, sarima_ci = run_sarima(train, n_total)
    logger.info(f"  SARIMA MAE={mean_absolute_error(y, sarima_fc[:n_test]):,.0f}  ({time.time()-t0:.1f}s)")

    # 2. Prophet
    prophet_fc, prophet_lo, prophet_hi = run_prophet(train, n_total, freq)
    logger.info(f"  Prophet MAE={mean_absolute_error(y, prophet_fc[:n_test]):,.0f}  ({time.time()-t0:.1f}s)")

    # 3. LSTM
    lstm_fc = run_lstm(train, n_total)
    logger.info(f"  LSTM MAE={mean_absolute_error(y, lstm_fc[:n_test]):,.0f}  ({time.time()-t0:.1f}s)")

    # 4. Ensemble 7 combinaisons
    s, p, l = sarima_fc[:n_test], prophet_fc[:n_test], lstm_fc[:n_test]
    combos = {
        "SARIMA":               s,
        "Prophet":              p,
        "LSTM":                 l,
        "SARIMA+Prophet":       (s + p) / 2,
        "SARIMA+LSTM":          (s + l) / 2,
        "Prophet+LSTM":         (p + l) / 2,
        "SARIMA+Prophet+LSTM":  (s + p + l) / 3,
    }
    maes    = {k: mean_absolute_error(y, v) for k, v in combos.items()}
    best    = min(maes, key=maes.get)
    best_mae = maes[best]

    # 5. Prévisions finales avec le meilleur modèle
    models_in = best.split("+")
    fc_list, lo_list, hi_list = [], [], []
    if "SARIMA"  in models_in:
        fc_list.append(sarima_fc)
        lo_list.append(sarima_ci[:, 0])
        hi_list.append(sarima_ci[:, 1])
    if "Prophet" in models_in:
        fc_list.append(prophet_fc)
        lo_list.append(prophet_lo)
        hi_list.append(prophet_hi)
    if "LSTM"    in models_in:
        fc_list.append(lstm_fc)
        lo_list.append(lstm_fc * 0.85)
        hi_list.append(lstm_fc * 1.15)

    best_fc = np.mean(fc_list, axis=0)
    best_lo = np.mean(lo_list, axis=0)
    best_hi = np.mean(hi_list, axis=0)

    # 6. Index de dates futures
    delta = pd.Timedelta(weeks=1) if freq == "W" else pd.DateOffset(months=1)
    future_idx = pd.date_range(
        start=train.index[-1] + delta,
        periods=n_total, freq=freq,
    )
    forecast_df = pd.DataFrame({
        "date":      future_idx.strftime("%Y-%m-%d"),
        "forecast":  best_fc.round(2),
        "lower_95":  best_lo.round(2),
        "upper_95":  best_hi.round(2),
    })

    # 7. Ranking
    ranking = [
        {"model": k, "mae": round(v, 2), "selected": k == best}
        for k, v in sorted(maes.items(), key=lambda x: x[1])
    ]

    logger.info(f"[Ensemble] Meilleur: {best} MAE={best_mae:,.0f}  total={time.time()-t0:.1f}s")

    return {
        "best_model":  best,
        "mae":         round(best_mae, 2),
        "forecast_df": forecast_df,
        "ranking":     ranking,
        "history":     train,
        "test":        test,
        "granularity": freq,
        "duration_s":  round(time.time() - t0, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
async def run_forecast(
    source_id: str,
    question:  str,
    rows:      list,
    granularity: str = "weekly",
    horizon:   int   = 12,
) -> dict:
    """
    Fonction principale appelée par l'endpoint /forecast de main.py.

    Params:
        source_id   : UUID de la source ERP
        question    : question NLU de l'utilisateur
        rows        : données SQL déjà extraites (liste de dicts)
        granularity : 'weekly' ou 'monthly'
        horizon     : nombre de périodes à prévoir

    Returns:
        dict avec best_model, mae, forecast_df, ranking, chart_data
    """
    freq = "W" if granularity == "weekly" else "MS"

    # Extraction de la série
    series = extract_time_series(rows, granularity)
    split  = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]

    # Entraînement dans un thread séparé (ne bloque pas la boucle async)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_ensemble(train, test, horizon, freq)
    )

    # Données pour le dashboard Chart.js
    history_list = [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for d, v in series.items()
    ]
    forecast_list = result["forecast_df"].to_dict(orient="records")

    result["chart_data"] = {
        "history":  history_list,
        "forecast": forecast_list,
        "ranking":  result["ranking"],
        "meta": {
            "best_model":  result["best_model"],
            "mae":         result["mae"],
            "granularity": granularity,
            "horizon":     horizon,
            "n_points":    len(series),
            "duration_s":  result["duration_s"],
        }
    }

    return result
