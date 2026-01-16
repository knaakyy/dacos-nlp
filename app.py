import os
import streamlit as st
import torch
import requests
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

# =============================
# 모델 다운로드 (HF)
# =============================
E4_URL = "https://huggingface.co/naakyy/kcelectra-e4/resolve/main/e4.bin"
BIN_PATH = "e4.bin"

def download_model_if_needed():
    if not os.path.exists(BIN_PATH):
        with st.spinner("🔽 모델 다운로드 중 (최초 1회)..."):
            r = requests.get(E4_URL, stream=True)
            r.raise_for_status()
            with open(BIN_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="한국어 변형 욕설 탐지 데모",
    page_icon="🛡️",
    layout="centered",
)

# 먼저 다운로드
download_model_if_needed()

# -----------------------------
# 경로/설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "beomi/KcELECTRA-base-v2022"
SAVED_MODEL_DIR = "E4_output/best_model"
MAX_LEN = 128

# -----------------------------
# 로딩 유틸
# -----------------------------
def _load_from_saved_dir(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def _load_from_bin(base_model_id: str, bin_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    config = AutoConfig.from_pretrained(base_model_id, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_id, config=config)

    state = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_artifacts():
    if os.path.isdir(SAVED_MODEL_DIR) and (
        os.path.isfile(os.path.join(SAVED_MODEL_DIR, "config.json"))
        or os.path.isfile(os.path.join(SAVED_MODEL_DIR, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(SAVED_MODEL_DIR, "model.safetensors"))
    ):
        return _load_from_saved_dir(SAVED_MODEL_DIR)

    if os.path.isfile(BIN_PATH):
        return _load_from_bin(BASE_MODEL_ID, BIN_PATH)

    raise FileNotFoundError(
        f"모델을 찾을 수 없습니다.\n- 폴더: {SAVED_MODEL_DIR}\n- 파일: {BIN_PATH}"
    )

def predict_proba_abusive(text: str, tokenizer, model) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    return float(probs[1].item())  # label 1 = abusive 가정

# -----------------------------
# UI 컴포넌트
# -----------------------------
def result_card(label: int):
    """label: 1=abusive, 0=non-abusive"""
    if label == 1:
        st.markdown(
            """
            <div style="
                background-color:#fdecea;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #e74c3c;
                font-size:20px;
                font-weight:600;
            ">
                🚨 판정 결과: <span style="color:#e74c3c;">욕설 (ABUSIVE)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background-color:#eafaf1;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #2ecc71;
                font-size:20px;
                font-weight:600;
            ">
                ✅ 판정 결과: <span style="color:#2ecc71;">정상 (NON-ABUSIVE)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

def prob_gauge(p: float):
    """0~1 확률 게이지"""
    st.markdown("### 📊 악성 확률")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(min(max(p, 0.0), 1.0))
    with col2:
        st.markdown(
            f"<div style='text-align:center; font-size:22px; font-weight:700;'>{p*100:.1f}%</div>",
            unsafe_allow_html=True,
        )

# -----------------------------
# 상단 타이틀
# -----------------------------
st.title("🛡️ 한국어 변형 욕설 탐지를 위한 자연어 처리 기반 분류 모델")
st.caption(
    "🔍 **문제의식**: 기존 욕설 필터링 시스템은 철자 변경·자음 분리·우회 표현 등 **변형 욕설**에 취약합니다.\n\n"
    "🧠 **접근 방식**: 본 프로젝트는 자연어 처리 기반 분석을 통해 문자 단위 변형에도 강건한 욕설 탐지 모델을 제안합니다."
)

# -----------------------------
# 사이드바: 페이지 선택
# -----------------------------
with st.sidebar:
    st.subheader("메뉴")
    page = st.radio("이동", ["데모", "프로젝트 소개"], index=0)

    st.divider()
    st.subheader("설정")
    threshold = st.slider("판정 임계값 (abusive)", 0.10, 0.90, 0.50, 0.05)
    st.write(f"Device: `{DEVICE}`")
    st.write(f"Max length: `{MAX_LEN}`")

# -----------------------------
# 프로젝트 소개 페이지
# -----------------------------
if page == "프로젝트 소개":
    st.markdown("""
## 1. 🧭 프로젝트 소개
본 프로젝트는 기존 필터링 시스템이 탐지하지 못하는 **변형 욕설**
(의도적으로 형태를 변형한 비속어)을 효과적으로 감지하는 것을 목표로 합니다.  
🧠 자연어 처리 기반 분석을 통해 **문자 단위 변형에도 강건한 욕설 탐지 모델**을 구축하고자 합니다.

## 2. ❗ 문제 정의
기존 욕설 필터링 시스템은 사전 기반 접근에 의존하는 경우가 많아  
✏️ 철자 변경 · 🔤 자음 분리 · 🌀 우회 표현 등 **변형된 욕설에 취약**합니다.  
이로 인해 온라인 커뮤니티 및 서비스 환경에서 부적절한 표현을 충분히 차단하지 못하는 문제가 발생합니다.

## 3. 🛠️ 사용 데이터 및 기술
- **📂 사용 데이터**: 욕설 및 비욕설 문장 데이터  
  (정상 표현 + 변형 욕설 포함)
- **⚙️ 기술 스택**:  
  텍스트 전처리, 서브워드/문자 단위 토큰화,  
  임베딩 기반 표현 학습, 머신러닝 · 딥러닝 분류 모델
- **💻 분석 환경**:  
  Python 기반 자연어 처리 프레임워크 활용

## 4. 🌱 결과 및 기대 효과
본 프로젝트는 단순 키워드 매칭을 넘어  
욕설의 **의미와 패턴을 학습하는 탐지 방식**을 제안합니다.  
이를 통해 온라인 플랫폼에서 욕설 필터링 정확도를 향상시키고,  
🤝 **건강한 커뮤니케이션 환경 조성**에 기여할 것으로 기대됩니다.

""")
    st.stop()

# -----------------------------
# 모델 로드
# -----------------------------
try:
    tokenizer, model = load_artifacts()
except Exception as e:
    st.error("모델 로딩 실패")
    st.code(str(e))
    st.stop()

# -----------------------------
# 데모 입력
# -----------------------------
text = st.text_area(
    "🔎 문장 입력",
    height=120,
    placeholder="예) ㅆㅣㅂㅏㄹ ㅋㅋ, ㅅㅂ 뭐함, @@@"
)

run = st.button("분석하기", type="primary", use_container_width=True)

# -----------------------------
# 결과 영역
# -----------------------------
if run:
    if not text.strip():
        st.warning("텍스트를 입력해줘!")
    else:
        p = predict_proba_abusive(text, tokenizer, model)
        pred = 1 if p >= threshold else 0

        st.subheader("결과")
        result_card(pred)

        st.divider()
        prob_gauge(p)

        with st.expander("🔍 자세히 보기"):
            st.write(f"- 적용 임계값: **{threshold:.2f}**")
            st.write(f"- p(abusive): **{p:.4f}**")
            st.write("- 참고: 모델은 오탐/미탐이 있을 수 있습니다.")

# -----------------------------
# 예시 블록 (항상 표시되게)
# -----------------------------
st.divider()
st.subheader("🧪 변형 욕설 예시 테스트")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**변형 욕설**")
    st.code("ㅆㅣㅂㅏㄹ ㅋㅋ")
    st.code("ㅅㅂ 뭐하냐")

with col2:
    st.markdown("**정상 표현**")
    st.code("ㅋㅋㅋㅋㅋㅋ")
    st.code("@@@")




