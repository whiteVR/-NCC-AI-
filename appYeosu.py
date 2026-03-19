import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 웹 페이지 기본 설정 ---
st.set_page_config(page_title="NCC 분해로 AI 모니터링", layout="wide")

st.title("🏭 에틸렌 분해로(NCC) AI 예지보전 및 자율 제어 대시보드")
st.markdown("실시간 센서 분석(Anomaly Detection) 및 잔여 수명(RUL) 예측 기반 자동 제어 루프(Control Loop)")

# --- 데이터 로드 ---
@st.cache_data
def load_data():
    return pd.read_csv('furnace_coking_data.csv')

df = load_data()

# --- 사이드바: 시뮬레이션 설정 ---
st.sidebar.header("⚙️ 시뮬레이션 설정")
current_step = st.sidebar.slider("현재 시점 (Time Step) 설정:", min_value=15, max_value=50, value=35)
critical_temp = st.sidebar.number_input("위험 한계 온도 (℃):", min_value=1000, max_value=1100, value=1060)

current_df = df[df['Time_Step'] <= current_step].copy()

# --- 1. 이상 탐지 (Isolation Forest) ---
X = current_df[['TMT_degC', 'Fuel_Flow_kgh']]
iso_model = IsolationForest(contamination=0.2, random_state=42)
current_df['Anomaly'] = iso_model.fit_predict(X)

# --- 2. 잔여 수명(RUL) 예측 (Linear Regression) ---
recent_df = current_df.tail(15)
lr_model = LinearRegression()
lr_model.fit(recent_df[['Time_Step']], recent_df['TMT_degC'])

weight = lr_model.coef_[0]
bias = lr_model.intercept_
predicted_failure_step = (critical_temp - bias) / weight
rul = predicted_failure_step - current_step
current_temp = current_df['TMT_degC'].iloc[-1]

# --- 화면 출력부 (상단 UI) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="현재 튜브 표면 온도 (TMT)", value=f"{current_temp:.1f} ℃", delta=f"한계까지 {critical_temp - current_temp:.1f} ℃ 남음", delta_color="inverse")
with col2:
    st.metric(label="예상 고장 시점", value=f"Step {predicted_failure_step:.1f}")
with col3:
    rul_status = "🚨 위험 (즉각 조치)" if rul <= 5 else ("⚠️ 경고 (부하 축소)" if rul <= 10 else "✅ 정상")
    st.metric(label=f"잔여 수명 (RUL) - {rul_status}", value=f"약 {rul:.1f} 스텝")

st.divider()

# --- 화면 출력부 (AI 자동 제어 시나리오) ---
st.subheader("🤖 스마트 팩토리 시스템 연동 및 자동 제어 시나리오 (Control Loop)")

if rul <= 5:
    st.error("🚨 **[위험 단계] 한계 온도 도달 임박! 파손 방지를 위한 비상 시퀀스를 가동합니다.**")
    st.markdown("""
    #### 1. 근본적 해결: 디코킹(Decoking) 모드 전환 시퀀스
    * **Phase A (원료 차단):** 미쓰비시 PLC로 제어 신호 송신 → 나프타(Feed) 주입 밸브 즉시 자동차단
    * **Phase B (스팀 주입):** 스팀 및 공기 주입 밸브 Open 제어 → 튜브 내부 잔류 탄소(Coking) 연소 시작
    * **Phase C (모니터링):** TMT 온도가 950℃ 이하로 안정화되는지 지속 추적
    
    #### 3. 스마트 팩토리 시스템과의 연동 (Control Loop)
    * **디지털 트윈(Unity 6):** 3D 설비 에셋 색상을 '적색 점멸(위험)'에서 '푸른색(디코킹 진행 중)'으로 자동 전환
    * **알람 시스템:** 중앙 관제실 및 현장 작업자 스마트폰으로 비상 알람 및 시퀀 가동 내역 푸시 알림
    """)

elif rul <= 15:
    st.warning("⚠️ **[경고 단계] 코킹 가속화 감지. 즉각적인 가동 중단이 불가할 경우 비상 조치를 권장합니다.**")
    st.markdown("""
    #### 2. 단기 비상 조치: 가동 부하 축소 (Load Reduction)
    * **조치 A (투입량 감소):** 나프타 투입량을 현재 대비 10~15% 하향 조정하여 설비 열 스트레스 완화 (PLC 제어 권고)
    * **조치 B (목표 온도 하향):** 출구 목표 온도(COT Set-point)를 830℃에서 820℃로 임시 하향 조정
    * **주의 사항:** 이는 파손을 지연시키는 조치이며, RUL 5스텝 도달 전 반드시 디코킹 일정을 확보해야 합니다.
    
    #### 3. 스마트 팩토리 시스템과의 연동 (Control Loop)
    * **생산 관리 연동:** MES(제조실행시스템)에 향후 5스텝 이내의 디코킹(생산 중단) 일정 조율 요청 자동 발송
    * **제어 상태:** AI가 최적의 부하 축소 비율을 계산하여 관리자의 '승인 대기' 상태로 대기 중
    """)

else:
    st.success("✅ **[정상 단계] 설비가 안정적으로 가동되고 있습니다.**")
    st.markdown("""
    #### 3. 스마트 팩토리 시스템과의 연동 (Control Loop)
    * **데이터 파이프라인:** 센서 데이터(TMT, COT, 유량)가 엣지 디바이스를 거쳐 AI 모델로 정상 수집 중
    * **예지보전 감시:** 선형 회귀(LR) 및 이상 탐지(Isolation Forest) 알고리즘이 백그라운드에서 실시간 분석 중
    * **디지털 트윈:** 3D 관제 화면에 설비가 '초록색(정상)'으로 동기화되어 표시됨
    """)

st.divider()

# --- 화면 출력부 (시각화) ---
st.subheader("📈 실시간 온도 추세 및 예측 그래프")
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(df['Time_Step'], df['TMT_degC'], color='lightgray', linestyle='--', label='Future True Path')
normal_data = current_df[current_df['Anomaly'] == 1]
ax.scatter(normal_data['Time_Step'], normal_data['TMT_degC'], color='blue', label='Normal', s=40)

anomaly_data = current_df[current_df['Anomaly'] == -1]
if not anomaly_data.empty:
    ax.scatter(anomaly_data['Time_Step'], anomaly_data['TMT_degC'], color='red', marker='X', label='Anomaly Detected', s=80)

future_steps = np.array([current_step, predicted_failure_step]).reshape(-1, 1)
future_temps = lr_model.predict(future_steps)
ax.plot(future_steps, future_temps, color='orange', linestyle='-', linewidth=2, label='AI Predicted Trend')

ax.axhline(y=critical_temp, color='red', linestyle=':', label='Critical Threshold (1060°C)')
ax.axvline(x=current_step, color='green', linestyle='-', alpha=0.5, label='Current Time Step')

ax.set_xlabel("Time Step")
ax.set_ylabel("TMT (°C)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- 하단 상세 데이터 표 (새로 추가된 부분) ---
st.divider()
st.subheader("📊 실시간 센서 로그 및 AI 분석 데이터")

with st.expander("상세 데이터 표 열기 (최근 15개 스텝 기준)"):
    # 가독성을 높이기 위해 데이터프레임 복사 및 포맷팅
    display_df = current_df.copy()
    
    # AI 판단 결과를 직관적인 텍스트와 이모지로 변환
    display_df['AI_판단_상태'] = display_df['Anomaly'].map({1: '🟢 정상', -1: '🔴 이상(Coking)'})
    
    # 화면에 보여줄 컬럼 순서 정리 및 불필요한 컬럼 제거
    display_df = display_df[['Time_Step', 'COT_degC', 'TMT_degC', 'Fuel_Flow_kgh', 'AI_판단_상태']]
    
    # 최근 15개 데이터만 역순(최신순)으로 정렬하여 표시, TMT 컬럼에 색상 하이라이트 적용
    st.dataframe(
        display_df.tail(15).iloc[::-1].style.highlight_max(axis=0, subset=['TMT_degC'], color='#ffcccc'), 
        use_container_width=True
    )