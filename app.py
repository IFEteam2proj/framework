"""
Black-Litterman Asset Allocation Web Application
Supports static optimization, multi-period walk-forward backtesting (2019-2020),
and final-period portfolio extraction.
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
    compute_portfolio_metrics,
    run_black_litterman_backtest,
    parse_date_from_string,
    parse_omega_var_from_string
)
from sample_data import (
    SAMPLE_ASSETS,
    get_sample_market_weights_df,
    get_sample_covariance_df,
    get_sample_views_df,
    create_blank_views_template,
    create_blank_market_template,
    generate_sample_monthly_returns,
    create_blank_monthly_returns_template
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="블랙-리터만 자산배분 & 백테스팅 플랫폼",
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
    st.session_state.assets = list(SAMPLE_ASSETS)
    st.session_state.weights_df = get_sample_market_weights_df()
    st.session_state.cov_df = get_sample_covariance_df()
    st.session_state.views_df = get_sample_views_df(st.session_state.assets)
    st.session_state.returns_df = generate_sample_monthly_returns(
        st.session_state.assets,
        st.session_state.cov_df.values
    )
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
if "returns_df" not in st.session_state:
    st.session_state.returns_df = generate_sample_monthly_returns(
        st.session_state.assets,
        st.session_state.cov_df.values
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def reset_to_sample_data():
    st.session_state.assets = list(SAMPLE_ASSETS)
    st.session_state.weights_df = get_sample_market_weights_df()
    st.session_state.cov_df = get_sample_covariance_df()
    st.session_state.views_df = get_sample_views_df(st.session_state.assets)
    st.session_state.returns_df = generate_sample_monthly_returns(
        st.session_state.assets,
        st.session_state.cov_df.values
    )
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
    if st.button("🔄 기본 5종 샘플 데이터로 복원", use_container_width=True):
        reset_to_sample_data()
        st.success("기본 샘플 데이터로 복원되었습니다.")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("📊 Black-Litterman 포트폴리오 최적화 & 월간 백테스팅 플랫폼")
st.caption(
    "기본 시장 데이터(공분산 및 벤치마크 가중치)를 고정한 상태에서, "
    "월별 견해(Views)와 실제 월간 수익률(2019-2020)을 바탕으로 시계열 백테스팅 성과를 분석하고, "
    "마지막 시점 기준의 최적 포트폴리오를 도출합니다."
)

# Main Navigation Tabs
tab_market, tab_views, tab_backtest, tab_final_port, tab_downloads = st.tabs([
    "1️⃣ 기본 데이터 (시장/자산/수익률)",
    "2️⃣ 견해 행렬 (Investor Views)",
    "3️⃣ 📈 월간 백테스팅 (2019~2020 성과)",
    "4️⃣ 🎯 최종 시점 기준 포트폴리오 결과",
    "5️⃣ 💾 템플릿 및 CSV 다운로드"
])


# =============================================================================
# TAB 1: 기본 데이터 설정 (시장 가중치 & 공분산 행렬 & 월간 수익률)
# =============================================================================
with tab_market:
    st.markdown("### 🏢 기본 시장 자산, 공분산 행렬 및 월간 수익률 설정")
    st.info("이곳에서 자산 목록, 시장 포트폴리오 가중치($w_{mkt}$), 공분산 행렬($\Sigma$), "
            "그리고 2019-2020 월간 수익률 데이터를 지정합니다.")

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
                    st.session_state.returns_df = generate_sample_monthly_returns(new_assets, st.session_state.cov_df.values)
                    st.success(f"시장 가중치 업로드 완료! ({len(st.session_state.assets)}개 자산 인식)")
                else:
                    st.error("CSV 파일에 'Asset'과 'MarketWeight' 컬럼이 포함되어야 합니다.")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    with col_m2:
        st.subheader("📤 공분산 행렬 (Covariance Matrix) 업로드")
        st.markdown("형식: 첫 열이 자산명인 $N \\times N$ CSV (또는 인덱스가 자산명)")
        cov_file = st.file_uploader(
            "공분산 행렬 CSV 파일 선택",
            type=["csv"],
            key="cov_uploader"
        )
        if cov_file is not None:
            try:
                uploaded_cov = pd.read_csv(cov_file)
                first_col = uploaded_cov.columns[0]
                if not pd.api.types.is_numeric_dtype(uploaded_cov[first_col]):
                    uploaded_cov = uploaded_cov.set_index(first_col)
                uploaded_cov = uploaded_cov.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                st.session_state.cov_df = uploaded_cov
                cov_assets = list(uploaded_cov.columns.astype(str))

                if list(st.session_state.assets) != cov_assets:
                    st.session_state.assets = cov_assets
                    st.session_state.weights_df = pd.DataFrame({
                        "Asset": cov_assets,
                        "MarketWeight": [round(1.0 / len(cov_assets), 6)] * len(cov_assets)
                    })
                    st.session_state.views_df = create_blank_views_template(cov_assets)
                    st.session_state.returns_df = generate_sample_monthly_returns(cov_assets, uploaded_cov.values)
                    st.info(f"💡 공분산 행렬의 {len(cov_assets)}개 자산에 맞추어 자산 목록과 벤치마크 가중치 및 월간수익률이 자동 동기화되었습니다.")

                st.success(f"공분산 행렬 업로드 완료! ({uploaded_cov.shape[0]}x{uploaded_cov.shape[1]})")
            except Exception as e:
                st.error(f"공분산 행렬을 읽는 중 오류가 발생했습니다: {e}")

    st.markdown("---")
    st.subheader("📅 2019~2020 월간 수익률 데이터 (백테스팅용)")
    st.caption("백테스팅 시 자산들의 월별 실제 실현수익률 데이터입니다. 업로드하지 않으면 공분산 기반 시뮬레이션 데이터가 적용됩니다.")

    col_ret_up, col_ret_info = st.columns([1, 1])
    with col_ret_up:
        returns_file = st.file_uploader(
            "월간 수익률 CSV 파일 업로드",
            type=["csv"],
            key="returns_uploader"
        )
        if returns_file is not None:
            try:
                uploaded_rets = pd.read_csv(returns_file)
                st.session_state.returns_df = uploaded_rets
                st.success(f"월간 수익률 파일 업로드 완료! (총 {len(uploaded_rets)}개 월간 기간)")
            except Exception as e:
                st.error(f"수익률 파일 로드 중 오류: {e}")

    with col_ret_info:
        st.markdown("**월간 수익률 툴**")
        if st.button("🎲 현재 자산 및 공분산 기반 샘플 월간수익률 재합성", use_container_width=True):
            st.session_state.returns_df = generate_sample_monthly_returns(
                st.session_state.assets,
                st.session_state.cov_df.values
            )
            st.success("2019-01 ~ 2020-11 샘플 월간 수익률이 새로 생성되었습니다.")
            st.rerun()

    # Previews
    st.markdown("---")
    st.subheader("📋 등록된 기본 데이터 확인")
    p_col1, p_col2 = st.columns([1, 1.3])
    with p_col1:
        st.markdown("**자산 목록 및 시장 벤치마크 가중치 ($w_{mkt}$)**")
        st.dataframe(st.session_state.weights_df, use_container_width=True, height=220)
    with p_col2:
        st.markdown("**2019~2020 월간 수익률 데이터 미리보기 (상위 5개월)**")
        st.dataframe(st.session_state.returns_df.head(5), use_container_width=True, height=220)


# =============================================================================
# TAB 2: 견해 행렬 (Investor Views)
# =============================================================================
with tab_views:
    st.markdown("### 💡 투자자 견해 행렬 (Investor Views, $P, Q, \Omega$)")
    st.info("시계열 견해(예: 2018-12부터 2020-11까지의 월별 Top5 vs Bottom5 견해) 파일인 `investor_views.csv`를 업로드하시면 "
            "3번 탭에서 월별 백테스팅을 즉시 수행하고, 마지막 시점 견해를 4번 탭의 최종 포트폴리오로 도출합니다.")

    current_assets = list(st.session_state.assets)
    missing_assets = [a for a in current_assets if a not in st.session_state.views_df.columns]
    if missing_assets:
        for a in missing_assets:
            st.session_state.views_df[a] = 0.0

    col_v_up, col_v_wiz = st.columns([1, 1])

    with col_v_up:
        st.subheader("📤 견해 행렬 CSV 파일 업로드")
        views_file = st.file_uploader(
            "견해 행렬 CSV 선택 (예: e:\\investor_views.csv)",
            type=["csv"],
            key="views_uploader"
        )
        if views_file is not None:
            try:
                uploaded_views = pd.read_csv(views_file)
                if "Target_Return" in uploaded_views.columns:
                    st.session_state.views_df = uploaded_views
                    st.success(f"견해 행렬 업로드 완료! (총 {len(uploaded_views)}개 월별 견해)")
                else:
                    st.error("CSV 파일에 최소한 'Target_Return' 컬럼이 있어야 합니다.")
            except Exception as e:
                st.error(f"견해 파일을 읽는 중 오류 발생: {e}")

    with col_v_wiz:
        st.subheader("➕ 견해 즉석 추가 (View Wizard)")
        with st.expander("원클릭 견해 추가 도구", expanded=False):
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
                        new_view[a] = 1.0 if a == long_asset else (-1.0 if a == short_asset else 0.0)
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
    st.subheader(f"📝 현재 등록된 견해 목록 (총 {len(st.session_state.views_df)}개 행)")
    st.caption("표 안에서 직접 값을 더블클릭하여 수정할 수 있습니다.")
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
        if st.button("💾 견해 편집사항 저장", use_container_width=True):
            st.session_state.views_df = edited_views
            st.success("견해 행렬이 저장되었습니다.")


# =============================================================================
# TAB 3: 월간 백테스팅 (2019-2020 성과 분석)
# =============================================================================
with tab_backtest:
    st.markdown("### 📈 2019 ~ 2020 월간 시계열 백테스팅 성과 분석")
    st.caption(
        "각 월말 시점의 롱-숏 견해(Views)를 반영하여 블랙-리터만 모델로 포트폴리오를 동적 리밸런싱했을 때, "
        "시장 벤치마크 포트폴리오 대비 성과가 어떻게 개선되는지 시뮬레이션합니다."
    )

    assets = list(st.session_state.assets)
    n_assets = len(assets)
    w_mkt_arr = pd.to_numeric(st.session_state.weights_df["MarketWeight"], errors="coerce").fillna(0.0).values
    if len(w_mkt_arr) != n_assets:
        w_mkt_arr = np.ones(n_assets) / n_assets
    if np.sum(w_mkt_arr) > 0:
        w_mkt_arr = w_mkt_arr / np.sum(w_mkt_arr)

    cov_mat = st.session_state.cov_df.reindex(index=assets, columns=assets).apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    if cov_mat.shape != (n_assets, n_assets):
        cov_mat = np.eye(n_assets) * 0.04

    # Run backtest
    try:
        bt_results = run_black_litterman_backtest(
            views_df=st.session_state.views_df,
            returns_df=st.session_state.returns_df,
            cov_matrix=cov_mat,
            assets=assets,
            market_weights=w_mkt_arr,
            delta=float(st.session_state.delta),
            tau=float(st.session_state.tau),
            allow_short=bool(st.session_state.allow_short),
            omega_method=st.session_state.omega_method,
            risk_free_rate=float(st.session_state.rf)
        )
        st.session_state.bt_results = bt_results

        perf_df = bt_results["performance_df"]
        metrics = bt_results["metrics"]

        if len(perf_df) > 0:
            # 1. Metric Cards
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric(
                    label="누적 수익률 (Total Return)",
                    value=f"{metrics['total_bl_ret']*100:.2f}%",
                    delta=f"{(metrics['total_bl_ret'] - metrics['total_mkt_ret'])*100:+.2f}%p vs BM"
                )
            with c2:
                st.metric(
                    label="연율화 수익률 (CAGR)",
                    value=f"{metrics['cagr_bl']*100:.2f}%",
                    delta=f"{(metrics['cagr_bl'] - metrics['cagr_mkt'])*100:+.2f}%p vs BM"
                )
            with c3:
                st.metric(
                    label="연율화 변동성",
                    value=f"{metrics['vol_bl']*100:.2f}%",
                    delta=f"{(metrics['vol_bl'] - metrics['vol_mkt'])*100:+.2f}%p vs BM",
                    delta_color="inverse"
                )
            with c4:
                st.metric(
                    label="샤프 지수 (Sharpe Ratio)",
                    value=f"{metrics['sharpe_bl']:.3f}",
                    delta=f"{metrics['sharpe_bl'] - metrics['sharpe_mkt']:+.3f}"
                )
            with c5:
                st.metric(
                    label="최대 낙폭 (MDD)",
                    value=f"{metrics['max_dd_bl']*100:.2f}%",
                    delta=f"{(metrics['max_dd_bl'] - metrics['max_dd_mkt'])*100:+.2f}%p",
                    delta_color="inverse"
                )

            st.markdown("---")

            # 2. Plots
            col_p1, col_p2 = st.columns([1.2, 1])

            with col_p1:
                st.subheader("📈 누적 수익률 추이 (Cumulative Return)")
                cum_plot_df = perf_df.copy()
                cum_plot_df["BL 포트폴리오 (%)"] = cum_plot_df["BL_CumRet"] * 100.0
                cum_plot_df["시장 벤치마크 (%)"] = cum_plot_df["Benchmark_CumRet"] * 100.0

                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=cum_plot_df["Return_Period"],
                    y=cum_plot_df["BL 포트폴리오 (%)"],
                    mode="lines+markers",
                    name="BL 동적 자산배분",
                    line=dict(color="#1f77b4", width=3)
                ))
                fig_cum.add_trace(go.Scatter(
                    x=cum_plot_df["Return_Period"],
                    y=cum_plot_df["시장 벤치마크 (%)"],
                    mode="lines",
                    name="시장 벤치마크",
                    line=dict(color="#7f7f7f", width=2, dash="dash")
                ))
                fig_cum.update_layout(
                    title="2019-2020 누적 수익률 비교 (단위: %)",
                    xaxis_title="리밸런싱 월",
                    yaxis_title="누적 수익률 (%)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_cum, use_container_width=True)

            with col_p2:
                st.subheader("📊 월별 초과수익률 (Alpha vs Benchmark)")
                excess_df = perf_df.copy()
                excess_df["초과수익률 (%)"] = excess_df["Excess_Return"] * 100.0
                excess_df["Color"] = np.where(excess_df["초과수익률 (%)"] >= 0, "Outperform (+)", "Underperform (-)")

                fig_exc = px.bar(
                    excess_df,
                    x="Return_Period",
                    y="초과수익률 (%)",
                    color="Color",
                    color_discrete_map={"Outperform (+)": "#2ca02c", "Underperform (-)": "#d62728"},
                    title="월별 초과수익률 (단위: %p)"
                )
                fig_exc.update_layout(
                    xaxis_title="월",
                    yaxis_title="초과수익률 (%p)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_exc, use_container_width=True)

            # 3. Monthly Detailed Backtest Table
            st.subheader("📋 월별 백테스팅 성과 내역")
            display_perf = perf_df.copy()
            display_perf["BL_Return"] = (display_perf["BL_Return"] * 100).round(2).astype(str) + "%"
            display_perf["Benchmark_Return"] = (display_perf["Benchmark_Return"] * 100).round(2).astype(str) + "%"
            display_perf["Excess_Return"] = (display_perf["Excess_Return"] * 100).round(2).astype(str) + "%p"
            display_perf["BL_CumRet"] = (display_perf["BL_CumRet"] * 100).round(2).astype(str) + "%"
            display_perf["Benchmark_CumRet"] = (display_perf["Benchmark_CumRet"] * 100).round(2).astype(str) + "%"
            display_perf["BL_Drawdown"] = (display_perf["BL_Drawdown"] * 100).round(2).astype(str) + "%"

            st.dataframe(
                display_perf[["View_Date", "Return_Period", "BL_Return", "Benchmark_Return", "Excess_Return", "BL_CumRet", "Benchmark_CumRet", "BL_Drawdown"]],
                use_container_width=True
            )

        else:
            st.warning("백테스팅을 실행할 수 있는 유효한 기간이 없습니다. 견해 행렬과 월간 수익률 데이터를 확인하세요.")

    except Exception as e:
        st.error(f"백테스팅 실행 중 오류가 발생했습니다: {e}")


# =============================================================================
# TAB 4: 최종 시점 기준 포트폴리오 결과
# =============================================================================
with tab_final_port:
    st.markdown("### 🎯 마지막 시점 기준 최종 결과 포트폴리오")
    st.caption("시계열 견해 데이터의 가장 마지막 시점(예: 2020-11-30) 견해를 기반으로 도출된 최적 자산배분 비중입니다.")

    assets = list(st.session_state.assets)
    n_assets = len(assets)
    w_mkt_arr = pd.to_numeric(st.session_state.weights_df["MarketWeight"], errors="coerce").fillna(0.0).values
    if len(w_mkt_arr) != n_assets:
        w_mkt_arr = np.ones(n_assets) / n_assets
    if np.sum(w_mkt_arr) > 0:
        w_mkt_arr = w_mkt_arr / np.sum(w_mkt_arr)

    cov_mat = st.session_state.cov_df.reindex(index=assets, columns=assets).apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    if cov_mat.shape != (n_assets, n_assets):
        cov_mat = np.eye(n_assets) * 0.04

    # Determine last view
    views_df = st.session_state.views_df.copy()
    valid_views = views_df[pd.to_numeric(views_df.get("Target_Return", 0), errors="coerce").abs() > 1e-9]

    if len(valid_views) > 0:
        last_view_row = valid_views.iloc[-1]
        last_date = parse_date_from_string(last_view_row.get("View_Description", ""))
        last_desc = str(last_view_row.get("View_Description", "Last View"))

        st.info(f"📌 **적용된 마지막 시점 견해**: `{last_desc}`")

        # Single view BL calculation for final period
        P_final = np.array([[float(last_view_row.get(a, 0.0)) if pd.notna(last_view_row.get(a, 0.0)) else 0.0 for a in assets]])
        Q_final = np.array([float(last_view_row["Target_Return"])])
        conf_final = np.array([float(last_view_row.get("Confidence", 0.5))])

        custom_omega = parse_omega_var_from_string(last_desc)
        if custom_omega is not None and custom_omega > 0:
            Omega_final = np.array([[custom_omega]])
        elif "신뢰도" in st.session_state.omega_method:
            Omega_final = calculate_confidence_omega(P_final, float(st.session_state.tau), cov_mat, conf_final)
        else:
            Omega_final = calculate_he_litterman_omega(P_final, float(st.session_state.tau), cov_mat)

        pi_returns = calculate_implied_returns(float(st.session_state.delta), cov_mat, w_mkt_arr)
        mu_bl_final, sigma_bl_final, _ = black_litterman_posterior(
            float(st.session_state.tau), cov_mat, pi_returns, P_final, Q_final, Omega_final
        )
        w_bl_final = optimize_portfolio(
            mu_bl_final, sigma_bl_final,
            delta=float(st.session_state.delta),
            allow_short=bool(st.session_state.allow_short)
        )

        final_mkt_metrics = compute_portfolio_metrics(w_mkt_arr, pi_returns, cov_mat, risk_free_rate=float(st.session_state.rf))
        final_bl_metrics = compute_portfolio_metrics(w_bl_final, mu_bl_final, sigma_bl_final, risk_free_rate=float(st.session_state.rf))

        # Top Metric Cards
        f1, f2, f3 = st.columns(3)
        with f1:
            st.metric(
                label="최종 BL 기대수익률",
                value=f"{final_bl_metrics['expected_return']*100:.2f}%",
                delta=f"{(final_bl_metrics['expected_return'] - final_mkt_metrics['expected_return'])*100:+.2f}%p vs 벤치마크"
            )
        with f2:
            st.metric(
                label="최종 포트폴리오 변동성",
                value=f"{final_bl_metrics['volatility']*100:.2f}%",
                delta=f"{(final_bl_metrics['volatility'] - final_mkt_metrics['volatility'])*100:+.2f}%p",
                delta_color="inverse"
            )
        with f3:
            st.metric(
                label="최종 샤프 지수",
                value=f"{final_bl_metrics['sharpe_ratio']:.3f}",
                delta=f"{final_bl_metrics['sharpe_ratio'] - final_mkt_metrics['sharpe_ratio']:+.3f}"
            )

        st.markdown("---")

        # Visualization
        col_f_c1, col_f_c2 = st.columns(2)
        with col_f_c1:
            st.subheader("⚖️ 최종 포트폴리오 비중 비교")
            final_weights_df = pd.DataFrame({
                "자산": assets,
                "시장 벤치마크 비중": w_mkt_arr * 100,
                "BL 최종 최적 비중": w_bl_final * 100
            })
            fig_w_final = px.bar(
                final_weights_df,
                x="자산",
                y=["시장 벤치마크 비중", "BL 최종 최적 비중"],
                barmode="group",
                labels={"value": "비중 (%)", "variable": "구분"},
                title="마지막 시점 자산별 비중 배분",
                color_discrete_map={"시장 벤치마크 비중": "#aec7e8", "BL 최종 최적 비중": "#2ca02c"}
            )
            fig_w_final.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_w_final, use_container_width=True)

        with col_f_c2:
            st.subheader("📊 최종 기대수익률 비교: 균형(Π) vs BL 사후(μ_BL)")
            final_ret_df = pd.DataFrame({
                "자산": assets,
                "시장 균형수익률 (Pi)": pi_returns * 100,
                "BL 사후수익률 (BL)": mu_bl_final * 100
            })
            fig_r_final = px.bar(
                final_ret_df,
                x="자산",
                y=["시장 균형수익률 (Pi)", "BL 사후수익률 (BL)"],
                barmode="group",
                labels={"value": "연간 기대수익률 (%)", "variable": "구분"},
                title="마지막 시점 기대수익률",
                color_discrete_map={"시장 균형수익률 (Pi)": "#7f7f7f", "BL 사후수익률 (BL)": "#1f77b4"}
            )
            fig_r_final.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_r_final, use_container_width=True)

        # Final Table
        st.subheader("📋 마지막 시점 기준 상세 포트폴리오 배분표")
        final_table = pd.DataFrame({
            "자산명": assets,
            "시장 벤치마크 비중 (%)": (w_mkt_arr * 100).round(2),
            "BL 최종 최적 비중 (%)": (w_bl_final * 100).round(2),
            "비중 변화 (Δw, %p)": ((w_bl_final - w_mkt_arr) * 100).round(2),
            "시장 균형수익률 (Π, %)": (pi_returns * 100).round(2),
            "BL 사후수익률 (μ_BL, %)": (mu_bl_final * 100).round(2),
            "수익률 변화 (Δμ, %p)": ((mu_bl_final - pi_returns) * 100).round(2)
        })
        st.dataframe(final_table, use_container_width=True)

        # Save to session
        st.session_state.final_portfolio_df = final_table
        st.session_state.final_sigma_df = pd.DataFrame(sigma_bl_final, index=assets, columns=assets)

    else:
        st.warning("유효한 견해가 등록되지 않았습니다.")


# =============================================================================
# TAB 5: 템플릿 및 CSV 다운로드
# =============================================================================
with tab_downloads:
    st.markdown("### 💾 결과 CSV 다운로드 및 표준 템플릿 제공")
    st.caption("백테스팅 시계열 결과 및 마지막 시점 최적 포트폴리오를 다운로드하거나, 표준 템플릿 서식을 내려받을 수 있습니다.")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.subheader("📥 산출 결과 다운로드")
        if "bt_results" in st.session_state and len(st.session_state.bt_results["performance_df"]) > 0:
            bt_csv = df_to_csv_bytes(st.session_state.bt_results["performance_df"])
            st.download_button(
                label="📄 월별 백테스팅 수익률 내역 (CSV)",
                data=bt_csv,
                file_name="black_litterman_backtest_performance.csv",
                mime="text/csv",
                use_container_width=True
            )

            bt_w_csv = df_to_csv_bytes(st.session_state.bt_results["weights_df"])
            st.download_button(
                label="📄 백테스팅 월별 자산 비중 시계열 (CSV)",
                data=bt_w_csv,
                file_name="black_litterman_backtest_weights_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        if "final_portfolio_df" in st.session_state:
            final_csv = df_to_csv_bytes(st.session_state.final_portfolio_df)
            st.download_button(
                label="🎯 마지막 시점 최종 포트폴리오 비중 결과 (CSV)",
                data=final_csv,
                file_name="black_litterman_final_portfolio.csv",
                mime="text/csv",
                use_container_width=True
            )

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

        # Template 4: Monthly returns
        sample_rets_csv = df_to_csv_bytes(st.session_state.returns_df)
        st.download_button(
            label="📁 2019-2020 월간 수익률 서식 템플릿 (CSV)",
            data=sample_rets_csv,
            file_name="template_monthly_returns.csv",
            mime="text/csv",
            use_container_width=True
        )
