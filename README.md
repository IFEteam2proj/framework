# 📈 블랙-리터만(Black-Litterman) 자산배분 웹 플랫폼

사용자가 직접 시장 파라미터와 공분산 행렬, 투자자 견해(Views)를 입력 및 CSV로 업로드하고, 실시간으로 최적 자산배분 비중과 기대수익률을 시각화 및 다운로드할 수 있는 대화형 웹 애플리케이션입니다.

---

## ✨ 핵심 기능

1. **기본 시장 데이터와 견해 행렬의 독립적 분리**
   - **기본 데이터 (Market Data)**: 자산 목록, 시장 벤치마크 가중치($w_{mkt}$), 공분산 행렬($\Sigma$), 위험회피계수($\lambda$), $\tau$, 무위험수익률($r_f$)을 세션에 한 번 등록해 두면 유지됩니다.
   - **견해 행렬 (Investor Views)**: 기본 시장 데이터는 그대로 유지한 채, 견해 행렬($P, Q, \Omega$)만 CSV 업로드 또는 UI 테이블에서 자유롭게 변경해가며 즉각적인 시뮬레이션 결과를 확인할 수 있습니다.
2. **다양한 입력 인터페이스**
   - CSV 파일 업로드 지원 (가중치, 공분산, 견해)
   - 웹 화면 상에서의 실시간 대화형 테이블 편집 (`st.data_editor`)
   - 직관적인 **원클릭 견해 생성 마법사 (View Wizard)**: 상대 전망 및 절대 전망을 클릭만으로 손쉽게 추가
3. **고도화된 불확실성($\Omega$) 처리 모델**
   - 투자자 신뢰도 백분율(Confidence %) 기반 자동 산출
   - He & Litterman (1999) 자동 추정 기법 지원
4. **인터랙티브 시각화 & 성과 지표**
   - 시장 균형수익률($\Pi$) vs BL 사후수익률($\mu_{BL}$) 바 차트
   - 벤치마크 비중($w_{mkt}$) vs BL 최적배분 비중($w_{BL}$) 바 차트
   - 포트폴리오 기대수익률, 변동성, 샤프 지수 메트릭 카드
   - 세부 수치 비교 분석표
5. **CSV 내보내기 & 표준 템플릿 제공**
   - 최종 포트폴리오 비중 및 기대수익률 결과 CSV 다운로드
   - 사후 공분산 행렬($\Sigma_{BL}$) CSV 다운로드
   - 업로드용 표준 CSV 서식 템플릿 다운로드

---

## 🚀 로컬 실행 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 스트림릿 앱 실행
```bash
streamlit run app.py
```
브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

---

## ☁️ 배포(Deploy) 가이드

### Streamlit Community Cloud 배포 (무료)
1. GitHub 저장소에 본 프로젝트 코드(`app.py`, `bl_engine.py`, `sample_data.py`, `requirements.txt`)를 푸시합니다.
2. [share.streamlit.io](https://share.streamlit.io)에 로그인합니다.
3. **New app** 버튼을 누르고 해당 저장소와 브랜치를 선택한 뒤 Main file path를 `app.py`로 지정하고 **Deploy**를 클릭합니다.

---

## 📁 CSV 업로드 파일 규격

### 1. 시장 가중치 (`template_market_weights.csv`)
| Asset | MarketWeight |
| :--- | :--- |
| SPY (미국 대형주) | 0.45 |
| EFA (선진국 주식) | 0.25 |
| TLT (미국 장기채) | 0.30 |

### 2. 공분산 행렬 (`template_covariance_matrix.csv`)
| Asset | SPY (미국 대형주) | EFA (선진국 주식) | TLT (미국 장기채) |
| :--- | :--- | :--- | :--- |
| SPY (미국 대형주) | 0.0324 | 0.0245 | -0.0045 |
| EFA (선진국 주식) | 0.0245 | 0.0361 | -0.0032 |
| TLT (미국 장기채) | -0.0045 | -0.0032 | 0.0196 |

### 3. 견해 행렬 (`template_investor_views.csv`)
| View_Description | Target_Return | Confidence | SPY (미국 대형주) | EFA (선진국 주식) | TLT (미국 장기채) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SPY가 EFA 대비 2.5% 초과 상승 | 0.025 | 0.65 | 1.0 | -1.0 | 0.0 |
| TLT 절대 기대수익률 4.0% | 0.040 | 0.50 | 0.0 | 0.0 | 1.0 |
