"""
Black-Litterman Model Mathematical Engine
Contains functions for market equilibrium reverse optimization,
Bayesian view combination, covariance estimation, and portfolio optimization.
"""

from typing import Optional, Tuple
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
