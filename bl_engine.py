"""
Black-Litterman Model Mathematical Engine
Contains functions for market equilibrium reverse optimization,
Bayesian view combination, covariance estimation, and portfolio optimization.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_implied_returns(
    delta: float,
    cov_matrix: np.ndarray,
    market_weights: np.ndarray
) -> np.ndarray:
    """
    Reverse optimization to calculate Implied Equilibrium Excess Returns (Pi).
    Pi = delta * Sigma * w_mkt
    """
    cov = np.asarray(cov_matrix, dtype=float)
    w_mkt = np.asarray(market_weights, dtype=float).reshape(-1, 1)
    pi = delta * np.dot(cov, w_mkt)
    return pi.flatten()


def calculate_he_litterman_omega(
    P: np.ndarray,
    tau: float,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Computes view uncertainty matrix Omega using the He & Litterman (1999) method:
    Omega = diag(P * (tau * Sigma) * P^T)
    """
    P = np.asarray(P, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    tau_cov = tau * cov
    omega_diag = np.diag(np.dot(np.dot(P, tau_cov), P.T))
    # Ensure strictly positive diagonal elements
    omega_diag = np.where(omega_diag <= 1e-12, 1e-6, omega_diag)
    return np.diag(omega_diag)


def calculate_confidence_omega(
    P: np.ndarray,
    tau: float,
    cov_matrix: np.ndarray,
    confidences: np.ndarray
) -> np.ndarray:
    """
    Calculates Omega based on investor confidence levels (0.0 < conf < 1.0).
    Omega_k = ((1 - conf_k) / conf_k) * (P * tau * Sigma * P^T)_k
    """
    P = np.asarray(P, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    conf = np.asarray(confidences, dtype=float)
    
    # Clip confidence to (0.001, 0.999) to prevent division by zero or infinite variance
    conf = np.clip(conf, 0.001, 0.999)
    tau_cov = tau * cov
    base_var = np.diag(np.dot(np.dot(P, tau_cov), P.T))
    
    # Scale uncertainty inversely with confidence
    omega_diag = ((1.0 - conf) / conf) * base_var
    omega_diag = np.where(omega_diag <= 1e-12, 1e-6, omega_diag)
    return np.diag(omega_diag)


def black_litterman_posterior(
    tau: float,
    cov_matrix: np.ndarray,
    pi: np.ndarray,
    P: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    Omega: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Black-Litterman posterior expected returns and covariance matrix.
    Uses the robust Woodbury matrix identity representation:
    mu_BL = Pi + tau * Sigma * P^T * [P * tau * Sigma * P^T + Omega]^-1 * (Q - P * Pi)
    Sigma_BL = (1 + tau) * Sigma - tau^2 * Sigma * P^T * [P * tau * Sigma * P^T + Omega]^-1 * P * Sigma
    
    Returns:
        (posterior_returns, posterior_cov, M_inv)
    """
    cov = np.asarray(cov_matrix, dtype=float)
    pi_vec = np.asarray(pi, dtype=float).reshape(-1, 1)
    n = cov.shape[0]

    # If no views provided, return prior distributions
    if P is None or Q is None or len(P) == 0 or len(Q) == 0:
        return pi_vec.flatten(), cov, tau * cov

    P_mat = np.asarray(P, dtype=float)
    Q_vec = np.asarray(Q, dtype=float).reshape(-1, 1)

    if Omega is None:
        Omega_mat = calculate_he_litterman_omega(P_mat, tau, cov)
    else:
        Omega_mat = np.asarray(Omega, dtype=float)

    tau_cov = tau * cov
    # intermediate term: P * (tau * Sigma) * P^T + Omega
    middle = np.dot(np.dot(P_mat, tau_cov), P_mat.T) + Omega_mat
    middle_inv = np.linalg.pinv(middle)

    # Posterior expected return
    view_diff = Q_vec - np.dot(P_mat, pi_vec)
    adjustment = np.dot(np.dot(np.dot(tau_cov, P_mat.T), middle_inv), view_diff)
    mu_bl = pi_vec + adjustment

    # Posterior uncertainty matrix M_inv
    M_inv = tau_cov - np.dot(np.dot(np.dot(tau_cov, P_mat.T), middle_inv), np.dot(P_mat, tau_cov))

    # Posterior full asset return covariance: Sigma + M_inv
    sigma_bl = cov + M_inv

    # Ensure symmetry
    sigma_bl = (sigma_bl + sigma_bl.T) / 2.0

    return mu_bl.flatten(), sigma_bl, M_inv


def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    delta: float = 2.5,
    allow_short: bool = False,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> np.ndarray:
    """
    Calculates optimal portfolio weights under mean-variance framework.
    If allow_short is True and no bounds, uses closed-form solution:
    w* = (1 / delta) * inv(Sigma) * mu, normalized to sum to 1.
    If long-only (constrained), uses scipy.optimize.minimize.
    """
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(mu)

    if allow_short and min_weight < 0:
        try:
            cov_inv = np.linalg.pinv(cov)
            raw_weights = (1.0 / delta) * np.dot(cov_inv, mu)
            # Normalize sum to 1 if sum is not zero
            w_sum = np.sum(raw_weights)
            if abs(w_sum) > 1e-8:
                return raw_weights / w_sum
            return raw_weights
        except np.linalg.LinAlgError:
            pass  # Fall back to numerical optimization

    # Quadratic utility objective: min 0.5 * delta * w^T Sigma w - w^T mu
    def objective(w):
        return 0.5 * delta * np.dot(np.dot(w, cov), w) - np.dot(w, mu)

    def objective_grad(w):
        return delta * np.dot(cov, w) - mu

    # Constraints: sum(w) = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    # Bounds per asset
    if not allow_short:
        bounds = [(max(0.0, min_weight), max_weight) for _ in range(n)]
    else:
        bounds = [(min_weight, max_weight) for _ in range(n)]

    w0 = np.ones(n) / n
    res = minimize(
        objective,
        w0,
        jac=objective_grad,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    if res.success:
        weights = res.x
        # Clean small precision inaccuracies
        weights = np.where(np.abs(weights) < 1e-6, 0.0, weights)
        weights = weights / np.sum(weights)
        return weights
    else:
        # Fallback to normalized initial weights
        return w0


def compute_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    rf: Optional[float] = None
) -> dict:
    """
    Calculates expected return, volatility, and Sharpe ratio for a given portfolio.
    """
    if rf is not None:
        risk_free_rate = rf

    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    port_return = float(np.dot(w, mu))
    port_variance = float(np.dot(np.dot(w, cov), w))
    port_volatility = float(np.sqrt(max(0.0, port_variance)))

    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 1e-8 else 0.0

    return {
        "expected_return": port_return,
        "volatility": port_volatility,
        "sharpe_ratio": sharpe
    }


def parse_date_from_string(s: str) -> Optional[str]:
    """Extracts YYYY-MM-DD or YYYY-MM from a string (e.g. '[2018-12-31]')."""
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(s))
    if match:
        return match.group(1)
    match_m = re.search(r'(\d{4}-\d{2})', str(s))
    if match_m:
        return match_m.group(1)
    return None


def parse_omega_var_from_string(s: str) -> Optional[float]:
    """Extracts custom Omega variance from description like '(Omega var: 0.001209)'."""
    import re
    match = re.search(r'Omega\s*var:\s*([0-9.eE+-]+)', str(s), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def run_black_litterman_backtest(
    views_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray,
    assets: List[str],
    market_weights: np.ndarray,
    delta: float = 2.5,
    tau: float = 0.05,
    allow_short: bool = False,
    omega_method: str = "신뢰도(Confidence) 기반",
    risk_free_rate: float = 0.02
) -> dict:
    """
    Executes a walk-forward monthly backtest of Black-Litterman model using time-series views and returns.
    Aligns each monthly view to the subsequent month's asset returns.
    Produces cumulative returns, drawdowns, risk-adjusted metrics, and the final period portfolio.
    """
    n_assets = len(assets)
    w_mkt = np.asarray(market_weights, dtype=float).flatten()
    if np.sum(w_mkt) > 0:
        w_mkt = w_mkt / np.sum(w_mkt)

    # 1. Clean returns dataframe
    ret_df = returns_df.copy()
    first_col = ret_df.columns[0]
    if not pd.api.types.is_numeric_dtype(ret_df[first_col]):
        ret_df = ret_df.set_index(first_col)
    ret_df.index = [str(idx).strip() for idx in ret_df.index]
    # Reindex to target assets
    ret_df = ret_df.reindex(columns=assets).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 2. Parse views rows
    views_clean = views_df.copy()
    parsed_views = []
    for idx, row in views_clean.iterrows():
        desc = str(row.get("View_Description", f"View_{idx}"))
        dt = parse_date_from_string(desc)
        target_ret = pd.to_numeric(row.get("Target_Return", 0.0), errors="coerce")
        conf = pd.to_numeric(row.get("Confidence", 0.5), errors="coerce")
        omega_custom = parse_omega_var_from_string(desc)

        # Vector P for this view
        p_row = np.array([float(row.get(a, 0.0)) if pd.notna(row.get(a, 0.0)) else 0.0 for a in assets], dtype=float)

        if pd.notna(target_ret) and abs(target_ret) > 1e-9:
            parsed_views.append({
                "index": idx,
                "date": dt if dt else f"Period_{idx}",
                "description": desc,
                "target_return": float(target_ret),
                "confidence": float(conf) if pd.notna(conf) else 0.5,
                "omega_custom": omega_custom,
                "P": p_row
            })

    if len(parsed_views) == 0:
        raise ValueError("유효한 견해(Target_Return이 0이 아닌 행)가 없습니다.")

    # 3. Step through views
    backtest_records = []
    weights_history = []
    
    return_dates = list(ret_df.index)
    n_ret_periods = len(return_dates)

    final_info = None

    for i, v_info in enumerate(parsed_views):
        P_i = v_info["P"].reshape(1, -1)
        Q_i = np.array([v_info["target_return"]])
        conf_i = np.array([v_info["confidence"]])

        # Construct Omega
        if v_info["omega_custom"] is not None and v_info["omega_custom"] > 0:
            Omega_i = np.array([[v_info["omega_custom"]]])
        elif "신뢰도" in omega_method:
            Omega_i = calculate_confidence_omega(P_i, tau, cov_matrix, conf_i)
        else:
            Omega_i = calculate_he_litterman_omega(P_i, tau, cov_matrix)

        # Implied prior returns
        pi = calculate_implied_returns(delta, cov_matrix, w_mkt)

        # Posterior
        mu_bl, sigma_bl, _ = black_litterman_posterior(tau, cov_matrix, pi, P_i, Q_i, Omega_i)

        # Optimal weights for this view period
        w_bl = optimize_portfolio(mu_bl, sigma_bl, delta=delta, allow_short=allow_short)

        # Determine matching return period
        # If views have dates and return dates correspond:
        matched_return_idx = None
        if i < n_ret_periods:
            matched_return_idx = i

        # If this is the last view, save as final portfolio
        if i == len(parsed_views) - 1:
            final_info = {
                "final_date": v_info["date"],
                "final_description": v_info["description"],
                "weights": w_bl,
                "mu_bl": mu_bl,
                "sigma_bl": sigma_bl,
                "pi": pi,
                "view_P": P_i,
                "view_Q": Q_i
            }

        # If there is a corresponding realized return period to test
        if matched_return_idx is not None and matched_return_idx < n_ret_periods:
            ret_date = return_dates[matched_return_idx]
            realized_asset_rets = ret_df.iloc[matched_return_idx].values

            bl_ret = float(np.dot(w_bl, realized_asset_rets))
            mkt_ret = float(np.dot(w_mkt, realized_asset_rets))
            excess_ret = bl_ret - mkt_ret

            backtest_records.append({
                "View_Date": v_info["date"],
                "Return_Period": ret_date,
                "BL_Return": bl_ret,
                "Benchmark_Return": mkt_ret,
                "Excess_Return": excess_ret
            })

            w_dict = {"Return_Period": ret_date}
            for a_idx, a_name in enumerate(assets):
                w_dict[a_name] = w_bl[a_idx]
            weights_history.append(w_dict)

    perf_df = pd.DataFrame(backtest_records)
    weights_df = pd.DataFrame(weights_history)

    if len(perf_df) > 0:
        perf_df["BL_CumRet"] = (1.0 + perf_df["BL_Return"]).cumprod() - 1.0
        perf_df["Benchmark_CumRet"] = (1.0 + perf_df["Benchmark_Return"]).cumprod() - 1.0

        # Drawdown calculation
        bl_cum_series = (1.0 + perf_df["BL_Return"]).cumprod()
        mkt_cum_series = (1.0 + perf_df["Benchmark_Return"]).cumprod()

        bl_peak = bl_cum_series.cummax()
        mkt_peak = mkt_cum_series.cummax()

        bl_drawdown = (bl_cum_series - bl_peak) / bl_peak
        mkt_drawdown = (mkt_cum_series - mkt_peak) / mkt_peak

        perf_df["BL_Drawdown"] = bl_drawdown
        perf_df["Benchmark_Drawdown"] = mkt_drawdown

        # Summary statistics
        n_months = len(perf_df)
        years = max(n_months / 12.0, 1.0 / 12.0)

        total_bl_ret = float(bl_cum_series.iloc[-1] - 1.0)
        total_mkt_ret = float(mkt_cum_series.iloc[-1] - 1.0)

        cagr_bl = float((1.0 + total_bl_ret) ** (1.0 / years) - 1.0) if total_bl_ret > -1.0 else -1.0
        cagr_mkt = float((1.0 + total_mkt_ret) ** (1.0 / years) - 1.0) if total_mkt_ret > -1.0 else -1.0

        vol_bl = float(perf_df["BL_Return"].std() * np.sqrt(12))
        vol_mkt = float(perf_df["Benchmark_Return"].std() * np.sqrt(12))

        sharpe_bl = (cagr_bl - risk_free_rate) / vol_bl if vol_bl > 1e-6 else 0.0
        sharpe_mkt = (cagr_mkt - risk_free_rate) / vol_mkt if vol_mkt > 1e-6 else 0.0

        max_dd_bl = float(bl_drawdown.min())
        max_dd_mkt = float(mkt_drawdown.min())

        win_rate = float((perf_df["Excess_Return"] > 0).mean())

        metrics = {
            "n_periods": n_months,
            "total_bl_ret": total_bl_ret,
            "total_mkt_ret": total_mkt_ret,
            "cagr_bl": cagr_bl,
            "cagr_mkt": cagr_mkt,
            "vol_bl": vol_bl,
            "vol_mkt": vol_mkt,
            "sharpe_bl": sharpe_bl,
            "sharpe_mkt": sharpe_mkt,
            "max_dd_bl": max_dd_bl,
            "max_dd_mkt": max_dd_mkt,
            "win_rate": win_rate
        }
    else:
        metrics = {}

    return {
        "performance_df": perf_df,
        "weights_df": weights_df,
        "metrics": metrics,
        "final_portfolio": final_info
    }
