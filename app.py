"""
Black-Litterman Asset Allocation Web Application
Built with Streamlit, Plotly, Pandas, and SciPy.
"""

import io
from typing import List, Tuple
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bl_engine import (
    calculate_implied_returns,
    calculate_he_litterman_omega,
    calculate_confidence_omega,
    black_litterman_posterior,
    optimize_portfolio,
    compute_portfolio_metrics
)
from sample_data import (
    SAMPLE_ASSETS,
    get_sample_market_weights_df,
    get_sample_covariance_df,
    get_sample_views_df,
    create_blank_views_template,
    create_blank_market_template
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="블랙-리터만 자산배분 플랫폼",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "market_loaded" not in st.session_state:
    # Initialize with default sample data
    st.session_state.assets = list(SAMPLE_ASSETS)
    st.session_state.weights_df = get_sample_market_weights_df()
    st.session_state.cov_df = get_sample_covariance_df()
    st.session_state.views_df = get_sample_views_df(st.session_state.assets)
    st.session_state.market_loaded = True

if "delta" not in st.session_state:
    st.session_state.delta = 2.5
if "tau" not in st.session_state:
    st.session_state.tau = 0.05
if "rf" not in st.session_state:
    st.session_state.rf = 0.02
if "allow_short" not in st.session_state:
    st.session_state.allow_short = False
if "omega_method" not in st.session_state:
    st.session_state.omega_method = "신뢰도(Confidence) 기반"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def reset_to_sample_data():
    st.session_state.assets = list(SAMPLE_ASSETS)
    st.session_state.weights_df = get_sample_market_weights_df()
    st.session_state.cov_df = get_sample_covariance_df()
    st.session_state.views_df = get_sample_views_df(st.session_state.assets)
    st.session_state.delta = 2.5
    st.session_state.tau = 0.05
    st.session_state.rf = 0.02
    st.session_state.allow_short = False
    st.session_state.market_loaded = True


def df_to_csv_bytes(df: pd.DataFrame, include_index: bool = False) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=include_index, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")


# -----------------------------------------------------------------------------
# Sidebar: Global Parameters & Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 모델 파라미터")
    st.markdown("블랙-리터만 모델의 거시 하이퍼파라미터 설정")

    st.subheader("1. 거시 변수")
    delta_input = st.number_input(
        "위험 회피 계수 (Risk Aversion, λ)",
        min_value=0.1,
        max_value=20.0,
        value=float(st.session_state.delta),
        step=0.1,
        help="시장 전체의 위험 감수 수준 (보통 2.0 ~ 4.0)"
    )
    st.session_state.delta = delta_input

    tau_input = st.number_input(
        "사전분포 불확실성 스케일 (Tau, τ)",
        min_value=0.001,
        max_value=1.0,
        value=float(st.session_state.tau),
        step=0.01,
        format="%.4f",
        help="사전 균형 기대수익률의 불확실성 척도 (보통 0.01 ~ 0.05)"
    )
    st.session_state.tau = tau_input

    rf_input = st.number_input(
        "무위험 수익률 (Risk-Free Rate, r_f)",
        min_value=0.0,
        max_value=0.20,
        value=float(st.session_state.rf),
        step=0.005,
        format="%.3f",
        help="샤프 지수 계산 및 초과수익률 변환에 사용되는 연율화 무위험 금리"
    )
    st.session_state.rf = rf_input

    st.subheader("2. 최적화 및 불확실성 설정")
    omega_method_choice = st.selectbox(
        "오차 행렬 (Omega, Ω) 설정 방식",
        ["신뢰도(Confidence) 기반", "He & Litterman (1999) 자동추정"],
        index=0 if st.session_state.omega_method == "신뢰도(Confidence) 기반" else 1,
        help="전망(Views)의 불확실성을 산출하는 계산 모델"
    )
    st.session_state.omega_method = omega_method_choice

    short_choice = st.checkbox(
        "공매도 허용 (Short Selling Allowed)",
        value=st.session_state.allow_short,
        help="체크 해제 시 가중치는 0 이상 1 이하(Long-Only)로 제한됩니다."
    )
    st.session_state.allow_short = short_choice

    st.markdown("---")
    if st.button("🔄 기본 샘플 데이터로 복원", use_container_width=True):
        reset_to_sample_data()
        st.success("기본 샘플 데이터로 복원되었습니다.")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("📊 Black-Litterman 포트폴리오 최적화 프레임워크")
st.caption(
    "기본 시장 데이터(공분산 및 벤치마크 가중치)를 유지한 상태에서, "
    "투자자의 주관적 견해(Views)를 자유롭게 변경하고 시나리오별 결과를 즉각 비교 분석할 수 있습니다."
)

# Main Navigation Tabs
tab_market, tab_views, tab_results, tab_downloads = st.tabs([
    "1️⃣ 기본 데이터 (시장/자산 행렬)",
    "2️⃣ 견해 행렬 (Investor Views)",
    "3️⃣ 최적화 결과 및 시각화",
    "4️⃣ 템플릿 및 CSV 다운로드"
])


# =============================================================================
# TAB 1: 기본 데이터 설정 (시장 가중치 & 공분산 행렬)
# =============================================================================
with tab_market:
    st.markdown("### 🏢 기본 시장 자산 및 공분산 행렬 설정")
    st.info("이곳에서 자산 목록, 시장 포트폴리오 가중치, 공분산 행렬을 지정합니다. "
            "한 번 설정해두면 견해(Views) 탭에서 견해만 바꿔가면서 시뮬레이션할 수 있습니다.")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.subheader("📤 시장 가중치 (Market Weights) 업로드")
        st.markdown("형식: `Asset`, `MarketWeight` 컬럼을 포함하는 CSV")
        weights_file = st.file_uploader(
            "시장 가중치 CSV 파일 선택",
            type=["csv"],
            key="weights_uploader"
        )
        if weights_file is not None:
            try:
                uploaded_weights = pd.read_csv(weights_file)
                if "Asset" in uploaded_weights.columns and "MarketWeight" in uploaded_weights.columns:
                    # Clean and normalize weights
                    uploaded_weights["MarketWeight"] = pd.to_numeric(uploaded_weights["MarketWeight"], errors="coerce").fillna(0.0)
                    total_w = uploaded_weights["MarketWeight"].sum()
                    if total_w > 0:
                        uploaded_weights["MarketWeight"] = uploaded_weights["MarketWeight"] / total_w
                    new_assets = list(uploaded_weights["Asset"].astype(str))
                    st.session_state.weights_df = uploaded_weights
                    st.session_state.assets = new_assets
                    if list(st.session_state.cov_df.columns) != new_assets:
                        st.session_state.cov_df = pd.DataFrame(
                            np.eye(len(new_assets)) * 0.04,
                            index=new_assets,
                            columns=new_assets
                        )
                    st.session_state.views_df = create_blank_views_template(new_assets)
                    st.success(f"시장 가중치 업로드 완료! ({len(st.session_state.assets)}개 자산 인식)")
                else:
                    st.error("CSV 파일에 'Asset'과 'MarketWeight' 컬럼이 포함되어야 합니다.")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    with col_m2:
        st.subheader("📤 공분산 행렬 (Covariance Matrix) 업로드")
        st.markdown("형식: 행/열 인덱스가 자산명인 $N \\times N$ CSV (또는 첫 컬럼이 자산명)")
        cov_file = st.file_uploader(
            "공분산 행렬 CSV 파일 선택",
            type=["csv"],
            key="cov_uploader"
        )
        if cov_file is not None:
            try:
                uploaded_cov = pd.read_csv(cov_file)
                # Check if first column contains asset names
                first_col = uploaded_cov.columns[0]
                if not pd.api.types.is_numeric_dtype(uploaded_cov[first_col]):
                    uploaded_cov = uploaded_cov.set_index(first_col)
                uploaded_cov = uploaded_cov.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                st.session_state.cov_df = uploaded_cov
                cov_assets = list(uploaded_cov.columns.astype(str))

                # If uploaded covariance matrix assets differ from current assets, synchronize
                if list(st.session_state.assets) != cov_assets:
                    st.session_state.assets = cov_assets
                    st.session_state.weights_df = pd.DataFrame({
                        "Asset": cov_assets,
                        "MarketWeight": [round(1.0 / len(cov_assets), 6)] * len(cov_assets)
                    })
                    st.session_state.views_df = create_blank_views_template(cov_assets)
                    st.info(f"💡 공분산 행렬의 {len(cov_assets)}개 자산에 맞추어 자산 목록과 가중치가 자동 동기화되었습니다.")

                st.success(f"공분산 행렬 업로드 완료! ({uploaded_cov.shape[0]}x{uploaded_cov.shape[1]})")
            except Exception as e:
                st.error(f"공분산 행렬을 읽는 중 오류가 발생했습니다: {e}")

    st.markdown("---")
    st.subheader("📋 현재 등록된 기본 시장 데이터 확인 및 직접 편집")

    col_view1, col_view2 = st.columns([1, 1.3])

    with col_view1:
        st.markdown("**자산 목록 및 시장 벤치마크 가중치 ($w_{mkt}$)**")
        edited_weights = st.data_editor(
            st.session_state.weights_df,
            num_rows="dynamic",
            use_container_width=True,
            key="weights_editor"
        )
        if st.button("가중치 편집 내용 반영", key="apply_weights_btn"):
            edited_weights["MarketWeight"] = pd.to_numeric(edited_weights["MarketWeight"], errors="coerce").fillna(0.0)
            tot = edited_weights["MarketWeight"].sum()
            if tot > 0:
                edited_weights["MarketWeight"] = edited_weights["MarketWeight"] / tot
            st.session_state.weights_df = edited_weights
            st.session_state.assets = list(edited_weights["Asset"].astype(str))
            
            # Synchronize covariance columns if dimensions don't match
            n_assets = len(st.session_state.assets)
            if st.session_state.cov_df.shape != (n_assets, n_assets):
                st.session_state.cov_df = pd.DataFrame(
                    np.eye(n_assets) * 0.04,
                    index=st.session_state.assets,
                    columns=st.session_state.assets
                )
            # Synchronize views table template
            st.session_state.views_df = create_blank_views_template(st.session_state.assets)
            st.success("시장 가중치가 갱신되었으며 견해 행렬이 새 자산 목록에 맞게 재동기화되었습니다.")
            st.rerun()

    with col_view2:
        st.markdown("**자산 수익률 공분산 행렬 ($\Sigma$, 연율화)**")
        # Ensure cov_df index and columns match
        cov_display = st.session_state.cov_df.copy()
        edited_cov = st.data_editor(
            cov_display,
            use_container_width=True,
            key="cov_editor"
        )
        if st.button("공분산 행렬 편집 내용 반영", key="apply_cov_btn"):
            st.session_state.cov_df = edited_cov.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            st.success("공분산 행렬이 반영되었습니다.")
            st.rerun()


# =============================================================================
# TAB 2: 견해 행렬 (Investor Views)
# =============================================================================
with tab_views:
    st.markdown("### 💡 투자자 견해 행렬 (Investor Views, $P, Q, \Omega$)")
    st.info("기본 시장 행렬은 고정된 채로 유지되며, **이곳의 견해만 바꾸면 즉각적으로 새로운 최적 자산배분 결과를 도출**할 수 있습니다.")

    current_assets = list(st.session_state.assets)
    
    # Check if views_df has asset columns that match current assets
    missing_assets = [a for a in current_assets if a not in st.session_state.views_df.columns]
    if missing_assets:
        for a in missing_assets:
            st.session_state.views_df[a] = 0.0

    col_v_up, col_v_wiz = st.columns([1, 1])

    with col_v_up:
        st.subheader("📤 견해 행렬 CSV 파일 업로드")
        st.caption("컬럼 구성: `View_Description`, `Target_Return`, `Confidence` 및 각 자산명 컬럼")
        views_file = st.file_uploader(
            "견해 행렬 CSV 선택",
            type=["csv"],
            key="views_uploader"
        )
        if views_file is not None:
            try:
                uploaded_views = pd.read_csv(views_file)
                # Check basic columns
                if "Target_Return" in uploaded_views.columns:
                    st.session_state.views_df = uploaded_views
                    st.success(f"견해 행렬 업로드 완료! (총 {len(uploaded_views)}개 견해)")
                else:
                    st.error("CSV 파일에 최소한 'Target_Return' 컬럼이 있어야 합니다.")
            except Exception as e:
                st.error(f"견해 파일을 읽는 중 오류 발생: {e}")

    with col_v_wiz:
        st.subheader("➕ 견해 즉석 추가 (View Wizard)")
        with st.expander("원클릭 견해 추가 도구", expanded=True):
            view_type = st.radio("견해 유형", ["상대 전망 (Relative: A가 B보다 초과 상승)", "절대 전망 (Absolute: A의 특정 수익률 기대)"], horizontal=True)
            
            if "상대" in view_type:
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    long_asset = st.selectbox("성과 우세 자산 (Long +1)", current_assets, key="long_asset")
                with col_w2:
                    short_asset = st.selectbox("성과 열세 자산 (Short -1)", [a for a in current_assets if a != long_asset], key="short_asset")
                
                target_ret_pct = st.number_input("초과 기대수익률 (%p)", value=2.0, step=0.5, key="rel_ret")
                conf_val = st.slider("견해 신뢰도 (Confidence %)", min_value=10, max_value=95, value=65, step=5, key="rel_conf")
                
                if st.button("상대 견해 추가하기", use_container_width=True):
                    new_view = {
                        "View_Description": f"{long_asset}이(가) {short_asset} 대비 {target_ret_pct:.1f}% 초과 상승",
                        "Target_Return": target_ret_pct / 100.0,
                        "Confidence": conf_val / 100.0
                    }
                    for a in current_assets:
                        if a == long_asset:
                            new_view[a] = 1.0
                        elif a == short_asset:
                            new_view[a] = -1.0
                        else:
                            new_view[a] = 0.0
                    
                    st.session_state.views_df = pd.concat([st.session_state.views_df, pd.DataFrame([new_view])], ignore_index=True)
                    st.success("상대 견해가 추가되었습니다!")
                    st.rerun()

            else:
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    target_asset = st.selectbox("대상 자산", current_assets, key="abs_asset")
                with col_w2:
                    abs_ret_pct = st.number_input("절대 기대수익률 (%)", value=7.0, step=0.5, key="abs_ret")
                
                conf_val = st.slider("견해 신뢰도 (Confidence %)", min_value=10, max_value=95, value=50, step=5, key="abs_conf")
                
                if st.button("절대 견해 추가하기", use_container_width=True):
                    new_view = {
                        "View_Description": f"{target_asset} 절대 기대수익률 {abs_ret_pct:.1f}%",
                        "Target_Return": abs_ret_pct / 100.0,
                        "Confidence": conf_val / 100.0
                    }
                    for a in current_assets:
                        new_view[a] = 1.0 if a == target_asset else 0.0
                    
                    st.session_state.views_df = pd.concat([st.session_state.views_df, pd.DataFrame([new_view])], ignore_index=True)
                    st.success("절대 견해가 추가되었습니다!")
                    st.rerun()

    st.markdown("---")
    st.subheader("📝 견해 행렬 ($P, Q$) 대화형 편집 테이블")
    st.caption("표 안에서 직접 값을 더블클릭하여 수정하거나 행을 추가/삭제할 수 있습니다. "
               "`Target_Return`은 소수점 형태(예: 0.03 = 3%), `Confidence`는 0~1 사이 값입니다.")

    # Ensure required columns
    base_cols = ["View_Description", "Target_Return", "Confidence"]
    full_cols = [c for c in base_cols if c in st.session_state.views_df.columns] + [a for a in current_assets if a in st.session_state.views_df.columns]
    
    views_edit_df = st.session_state.views_df[full_cols].copy()
    
    edited_views = st.data_editor(
        views_edit_df,
        num_rows="dynamic",
        use_container_width=True,
        key="views_table_editor"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("💾 견해 변경사항 적용", use_container_width=True):
            st.session_state.views_df = edited_views
            st.success("견해 행렬이 성공적으로 저장되었습니다. '결과 및 시각화' 탭에서 분석을 확인하세요.")
    with col_btn2:
        if st.button("🗑️ 모든 견해 초기화 (비우기)", use_container_width=False):
            st.session_state.views_df = create_blank_views_template(current_assets)
            st.info("모든 견해가 비워졌습니다. 이제 순수 시장 균형 포트폴리오 상태가 됩니다.")
            st.rerun()


# =============================================================================
# TAB 3: 최적화 결과 및 시각화
# =============================================================================
with tab_results:
    st.markdown("### 🚀 블랙-리터만 포트폴리오 산출 결과")

    assets = list(st.session_state.assets)
    n_assets = len(assets)

    # 1. Prepare Market Weights and Covariance
    w_mkt_series = pd.to_numeric(st.session_state.weights_df["MarketWeight"], errors="coerce").fillna(0.0).values
    if len(w_mkt_series) != n_assets:
        w_mkt_series = np.ones(n_assets) / n_assets
    if np.sum(w_mkt_series) > 0:
        w_mkt_series = w_mkt_series / np.sum(w_mkt_series)

    # Covariance Matrix alignment
    cov_mat = st.session_state.cov_df.reindex(index=assets, columns=assets).apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    if cov_mat.shape != (n_assets, n_assets):
        cov_mat = np.eye(n_assets) * 0.04

    # 2. Compute Implied Equilibrium Returns (Pi)
    delta = float(st.session_state.delta)
    tau = float(st.session_state.tau)
    rf = float(st.session_state.rf)
    allow_short = bool(st.session_state.allow_short)

    pi_returns = calculate_implied_returns(delta, cov_mat, w_mkt_series)

    # 3. Extract P, Q, and Omega from views_df
    views_df = st.session_state.views_df.copy()
    valid_views = []
    
    if "Target_Return" in views_df.columns:
        for idx, row in views_df.iterrows():
            q_val = pd.to_numeric(row.get("Target_Return", 0.0), errors="coerce")
            if pd.notna(q_val) and abs(q_val) > 1e-9:
                valid_views.append(row)

    if len(valid_views) > 0:
        clean_views_df = pd.DataFrame(valid_views)
        Q = clean_views_df["Target_Return"].astype(float).values
        confidences = clean_views_df["Confidence"].astype(float).values if "Confidence" in clean_views_df.columns else np.full(len(Q), 0.5)

        # Build P matrix
        P_list = []
        for _, row in clean_views_df.iterrows():
            row_p = [float(row.get(a, 0.0)) if pd.notna(row.get(a, 0.0)) else 0.0 for a in assets]
            P_list.append(row_p)
        P = np.array(P_list, dtype=float)

        # Calculate Omega
        if st.session_state.omega_method == "신뢰도(Confidence) 기반":
            Omega = calculate_confidence_omega(P, tau, cov_mat, confidences)
        else:
            Omega = calculate_he_litterman_omega(P, tau, cov_mat)

        # Posterior calculation
        mu_bl, sigma_bl, M_inv = black_litterman_posterior(tau, cov_mat, pi_returns, P, Q, Omega)
        has_views = True
    else:
        mu_bl = pi_returns.copy()
        sigma_bl = cov_mat.copy()
        has_views = False
        st.warning("⚠️ 활성화된 견해(Views)가 없습니다. 시장 균형 상태(Prior)로 계산됩니다.")

    # 4. Optimize Portfolios
    # Market equilibrium portfolio optimization
    w_prior_opt = optimize_portfolio(pi_returns, cov_mat, delta=delta, allow_short=allow_short)
    # BL posterior portfolio optimization
    w_bl_opt = optimize_portfolio(mu_bl, sigma_bl, delta=delta, allow_short=allow_short)

    # 5. Compute Metrics
    mkt_metrics = compute_portfolio_metrics(w_mkt_series, pi_returns, cov_mat, risk_free_rate=rf)
    bl_metrics = compute_portfolio_metrics(w_bl_opt, mu_bl, sigma_bl, risk_free_rate=rf)

    # Display Top Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="BL 포트폴리오 기대수익률",
            value=f"{bl_metrics['expected_return']*100:.2f}%",
            delta=f"{(bl_metrics['expected_return'] - mkt_metrics['expected_return'])*100:+.2f}%p vs 벤치마크"
        )
    with m2:
        st.metric(
            label="BL 포트폴리오 변동성 (위험)",
            value=f"{bl_metrics['volatility']*100:.2f}%",
            delta=f"{(bl_metrics['volatility'] - mkt_metrics['volatility'])*100:+.2f}%p vs 벤치마크",
            delta_color="inverse"
        )
    with m3:
        st.metric(
            label="BL 샤프 지수 (Sharpe Ratio)",
            value=f"{bl_metrics['sharpe_ratio']:.3f}",
            delta=f"{bl_metrics['sharpe_ratio'] - mkt_metrics['sharpe_ratio']:+.3f}"
        )
    with m4:
        st.metric(
            label="반영된 견해 수 (Views Count)",
            value=f"{len(valid_views)}개",
            delta="신뢰도 기반 결합" if st.session_state.omega_method.startswith("신뢰도") else "He-Litterman 결합"
        )

    st.markdown("---")

    # Visualizations Section
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("📊 자산별 기대수익률 비교: 균형(Π) vs 사후(μ_BL)")
        ret_df = pd.DataFrame({
            "자산": assets,
            "시장 균형 기대수익률 (Pi)": pi_returns * 100,
            "블랙-리터만 사후수익률 (BL)": mu_bl * 100
        })
        fig_ret = px.bar(
            ret_df,
            x="자산",
            y=["시장 균형 기대수익률 (Pi)", "블랙-리터만 사후수익률 (BL)"],
            barmode="group",
            labels={"value": "연간 기대수익률 (%)", "variable": "구분"},
            title="기대수익률 변화 비교 (단위: %)",
            color_discrete_map={
                "시장 균형 기대수익률 (Pi)": "#7f7f7f",
                "블랙-리터만 사후수익률 (BL)": "#1f77b4"
            }
        )
        fig_ret.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_chart2:
        st.subheader("⚖️ 포트폴리오 가중치 비교: 벤치마크 vs BL 최적배분")
        weights_comparison = pd.DataFrame({
            "자산": assets,
            "시장 벤치마크 비중": w_mkt_series * 100,
            "BL 최적 비중": w_bl_opt * 100,
            "가중치 변동 (Δw)": (w_bl_opt - w_mkt_series) * 100
        })
        fig_w = px.bar(
            weights_comparison,
            x="자산",
            y=["시장 벤치마크 비중", "BL 최적 비중"],
            barmode="group",
            labels={"value": "포트폴리오 비중 (%)", "variable": "구분"},
            title="자산배분 비중 비교 (단위: %)",
            color_discrete_map={
                "시장 벤치마크 비중": "#aec7e8",
                "BL 최적 비중": "#2ca02c"
            }
        )
        fig_w.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_w, use_container_width=True)

    # Detailed Comparison Table
    st.subheader("📋 세부 비교 분석표")
    detailed_table = pd.DataFrame({
        "자산명": assets,
        "시장 벤치마크 비중 (%)": (w_mkt_series * 100).round(2),
        "BL 최적 비중 (%)": (w_bl_opt * 100).round(2),
        "비중 변화 (Δw, %p)": ((w_bl_opt - w_mkt_series) * 100).round(2),
        "시장 균형수익률 (Π, %)": (pi_returns * 100).round(2),
        "BL 사후수익률 (μ_BL, %)": (mu_bl * 100).round(2),
        "수익률 변화 (Δμ, %p)": ((mu_bl - pi_returns) * 100).round(2),
        "개별 변동성 (σ, %)": (np.sqrt(np.diag(cov_mat)) * 100).round(2)
    })
    st.dataframe(detailed_table, use_container_width=True)

    # Store calculated results in session for export
    st.session_state.result_summary = detailed_table
    st.session_state.posterior_cov_df = pd.DataFrame(sigma_bl, index=assets, columns=assets)


# =============================================================================
# TAB 4: 템플릿 및 CSV 다운로드
# =============================================================================
with tab_downloads:
    st.markdown("### 💾 결과 CSV 다운로드 및 표준 템플릿 제공")
    st.caption("최적화 결과 데이터를 다운로드하거나, 모델 업로드에 사용할 수 있는 표준 CSV 서식을 내려받을 수 있습니다.")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.subheader("📥 산출 결과 다운로드")
        if "result_summary" in st.session_state:
            res_csv = df_to_csv_bytes(st.session_state.result_summary)
            st.download_button(
                label="📄 포트폴리오 가중치 및 수익률 결과 (CSV)",
                data=res_csv,
                file_name="black_litterman_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            cov_res_csv = df_to_csv_bytes(st.session_state.posterior_cov_df, include_index=True)
            st.download_button(
                label="📄 사후 공분산 행렬 (Σ_BL) (CSV)",
                data=cov_res_csv,
                file_name="black_litterman_posterior_cov.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("먼저 '최적화 결과 및 시각화' 탭을 실행하면 다운로드 링크가 활성화됩니다.")

    with col_dl2:
        st.subheader("📥 입력 양식 표준 템플릿 다운로드")
        # Template 1: Market weights
        sample_w_csv = df_to_csv_bytes(st.session_state.weights_df)
        st.download_button(
            label="📁 시장 가중치 서식 템플릿 (CSV)",
            data=sample_w_csv,
            file_name="template_market_weights.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Template 2: Covariance
        sample_cov_csv = df_to_csv_bytes(st.session_state.cov_df, include_index=True)
        st.download_button(
            label="📁 공분산 행렬 서식 템플릿 (CSV)",
            data=sample_cov_csv,
            file_name="template_covariance_matrix.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Template 3: Views
        sample_views_csv = df_to_csv_bytes(st.session_state.views_df)
        st.download_button(
            label="📁 견해 행렬(Views) 서식 템플릿 (CSV)",
            data=sample_views_csv,
            file_name="template_investor_views.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📖 CSV 파일 작성 가이드")
    with st.expander("CSV 형식 안내 보기", expanded=True):
        st.markdown("""
        * **1. 시장 가중치 (Market Weights)**:
          - 필수 컬럼: `Asset` (자산명), `MarketWeight` (시가총액 비중 또는 시가총액 금액).
          - 합계가 1이 아니더라도 자동으로 합계가 1이 되도록 정규화됩니다.
        * **2. 공분산 행렬 (Covariance Matrix)**:
          - $N \\times N$ 정방행렬 형태입니다.
          - 행 인덱스와 열 헤더에 동일한 자산명이 입력되어야 합니다.
        * **3. 견해 행렬 (Investor Views)**:
          - 필수 컬럼: `View_Description`, `Target_Return`, `Confidence` 및 **각 자산명 컬럼**.
          - **상대 전망**: 우세 자산 컬럼에 `1.0`, 열세 자산 컬럼에 `-1.0` 입력.
          - **절대 전망**: 대상 자산 컬럼에 `1.0`, 나머지 자산 컬럼에 `0.0` 입력.
          - `Target_Return`: 연간 초과 기대수익률 (예: 2.5% = `0.025`).
          - `Confidence`: 신뢰도 비율 (예: 65% = `0.65`).
        """)
