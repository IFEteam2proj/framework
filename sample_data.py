"""
Sample data generator and template provider for the Black-Litterman model.
Provides standard asset universe, covariance matrix, market weights,
and sample investor views.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


SAMPLE_ASSETS = ["SPY (미국 대형주)", "EFA (선진국 주식)", "EEM (신흥국 주식)", "TLT (미국 장기채)", "GLD (금)"]

SAMPLE_MARKET_WEIGHTS = {
    "SPY (미국 대형주)": 0.45,
    "EFA (선진국 주식)": 0.25,
    "EEM (신흥국 주식)": 0.10,
    "TLT (미국 장기채)": 0.15,
    "GLD (금)": 0.05
}

# Annualized covariance matrix for the 5 assets
SAMPLE_COV_DATA = [
    [0.0324, 0.0245, 0.0280, -0.0045, 0.0030],
    [0.0245, 0.0361, 0.0315, -0.0032, 0.0040],
    [0.0280, 0.0315, 0.0576, -0.0020, 0.0075],
    [-0.0045, -0.0032, -0.0020, 0.0196, 0.0025],
    [0.0030, 0.0040, 0.0075, 0.0025, 0.0256]
]


def get_sample_market_weights_df() -> pd.DataFrame:
    """Returns a DataFrame of sample assets and their market weights."""
    df = pd.DataFrame(
        list(SAMPLE_MARKET_WEIGHTS.items()),
        columns=["Asset", "MarketWeight"]
    )
    return df


def get_sample_covariance_df() -> pd.DataFrame:
    """Returns a DataFrame of sample covariance matrix."""
    df = pd.DataFrame(
        SAMPLE_COV_DATA,
        index=SAMPLE_ASSETS,
        columns=SAMPLE_ASSETS
    )
    return df


def get_sample_views_df(assets: List[str] = None) -> pd.DataFrame:
    """
    Returns a sample Views DataFrame.
    Columns: View_Description, Target_Return (Q), Confidence (0~1 or 0~100%), and asset weight columns (P matrix)
    """
    if assets is None:
        assets = SAMPLE_ASSETS

    # View 1: SPY will outperform EFA by 2.5% (Relative view)
    # View 2: EEM will have an absolute return of 8.0% (Absolute view)
    # View 3: GLD will outperform TLT by 1.5% (Relative view)
    views_data = []

    # View 1
    row1 = {"View_Description": "미국 대형주(SPY)가 선진국(EFA) 대비 2.5% 초과 상승", "Target_Return": 0.025, "Confidence": 0.65}
    for a in assets:
        if "SPY" in a:
            row1[a] = 1.0
        elif "EFA" in a:
            row1[a] = -1.0
        else:
            row1[a] = 0.0
    views_data.append(row1)

    # View 2
    row2 = {"View_Description": "신흥국 주식(EEM) 절대 기대수익률 8.0%", "Target_Return": 0.080, "Confidence": 0.50}
    for a in assets:
        if "EEM" in a:
            row2[a] = 1.0
        else:
            row2[a] = 0.0
    views_data.append(row2)

    # View 3
    row3 = {"View_Description": "금(GLD)이 미국 장기채(TLT) 대비 1.5% 초과 상승", "Target_Return": 0.015, "Confidence": 0.70}
    for a in assets:
        if "GLD" in a:
            row3[a] = 1.0
        elif "TLT" in a:
            row3[a] = -1.0
        else:
            row3[a] = 0.0
    views_data.append(row3)

    cols = ["View_Description", "Target_Return", "Confidence"] + assets
    return pd.DataFrame(views_data)[cols]


def create_blank_views_template(assets: List[str]) -> pd.DataFrame:
    """Creates an empty template DataFrame for user views matching the current assets."""
    cols = ["View_Description", "Target_Return", "Confidence"] + assets
    empty_data = [{col: (0.0 if col != "View_Description" else "새 견해 예시") for col in cols}]
    return pd.DataFrame(empty_data)


def create_blank_market_template(n_assets: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates blank template DataFrames for market weights and covariance."""
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    weights_df = pd.DataFrame({
        "Asset": asset_names,
        "MarketWeight": [round(1.0 / n_assets, 4)] * n_assets
    })
    cov_matrix = np.eye(n_assets) * 0.04
    cov_df = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
    return weights_df, cov_df


def generate_sample_monthly_returns(
    assets: List[str],
    cov_matrix: np.ndarray = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic synthetic 2019-2020 monthly returns matching the user's backtest period.
    From 2019-01-31 to 2020-11-30 (23 months).
    """
    dates = [
        "2019-01-31", "2019-02-28", "2019-03-29", "2019-04-30", "2019-05-31", "2019-06-28",
        "2019-07-31", "2019-08-30", "2019-09-30", "2019-10-31", "2019-11-29", "2019-12-31",
        "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29", "2020-06-30",
        "2020-07-31", "2020-08-31", "2020-09-30", "2020-10-30", "2020-11-30"
    ]
    n_months = len(dates)
    n_assets = len(assets)

    rng = np.random.default_rng(seed)

    if cov_matrix is not None and cov_matrix.shape == (n_assets, n_assets):
        # Monthly covariance = annual covariance / 12
        monthly_cov = (cov_matrix + cov_matrix.T) / 2.0 / 12.0
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(monthly_cov)
        eigenvals = np.maximum(eigenvals, 1e-6)
        monthly_cov = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
        mean_annual_return = 0.08  # 8% annual
        mean_monthly = np.full(n_assets, mean_annual_return / 12.0)
        returns_data = rng.multivariate_normal(mean_monthly, monthly_cov, size=n_months)
    else:
        # Default ~15% annual vol -> 15%/sqrt(12) ~ 4.3% monthly vol
        returns_data = rng.normal(loc=0.007, scale=0.045, size=(n_months, n_assets))

    df = pd.DataFrame(returns_data, index=dates, columns=assets)
    df.index.name = "Date"
    return df.round(6).reset_index()


def create_blank_monthly_returns_template(assets: List[str]) -> pd.DataFrame:
    """Creates a blank monthly returns template with 2019-01 to 2020-11 dates."""
    dates = [
        "2019-01-31", "2019-02-28", "2019-03-29", "2019-04-30", "2019-05-31", "2019-06-28",
        "2019-07-31", "2019-08-30", "2019-09-30", "2019-10-31", "2019-11-29", "2019-12-31",
        "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29", "2020-06-30",
        "2020-07-31", "2020-08-31", "2020-09-30", "2020-10-30", "2020-11-30"
    ]
    df = pd.DataFrame({"Date": dates})
    for a in assets:
        df[a] = 0.01  # placeholder 1%
    return df

