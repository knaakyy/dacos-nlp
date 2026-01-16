import os
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="E4 악성댓글 탐지 (KC-ELECTRA)",
    page_icon="🛡️",
    layout="centered",
)

# -----------------------------
# 경로/설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ ipynb에서 사용한 모델 ID와 맞춤
BASE_MODEL_ID = "beomi/KcELECTRA-base-v2022"

# 1) (권장) save_pretrained 폴더가 있으면 그걸 사용
SAVED_MODEL_DIR = "E4_output/best_model"   # 레포에 이 폴더째 올리면 제일 편함

# 2) (대안) state_dict만 있을 때 (e4.bin 하나만 있을 때)
BIN_PATH = "e4.bin"  # 레포 루트에 e4.bin 두는 기준. 다른 위치면 경로만 수정.

MAX_LEN = 128  # ipynb에서 max_length=128로 학습
LABEL_MAP = {0: "NON-ABUSIVE", 1: "ABUSIVE"}  # 너희 라벨 정의 기준

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
    """
    1) E4_output/best_model 폴더가 있으면 우선 로드
    2) 없으면 e4.bin(state_dict) 로드
    """
    if os.path.isdir(SAVED_MODEL_DIR) and (
        os.path.isfile(os.path.join(SAVED_MODEL_DIR, "config.json"))
        or os.path.isfile(os.path.join(SAVED_MODEL_DIR, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(SAVED_MODEL_DIR, "model.safetensors"))
    ):
        return _load_from_saved_dir(SAVED_MODEL_DIR)

    if os.path.isfile(BIN_PATH):
        return _load_from_bin(BASE_MODEL_ID, BIN_PATH)

    raise FileNotFoundError(
        f"모델을 찾을 수 없습니다.\n"
        f"- 폴더: {SAVED_MODEL_DIR}\n"
        f"- 파일: {BIN_PATH}\n"
        f"둘 중 하나를 레포에 포함시켜주세요."
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

    # label 1 = abusive 가정
    return float(probs[1].item())

# -----------------------------
# UI
# -----------------------------
st.title("🛡️ E4 악성댓글 탐지 데모 (KC-ELECTRA)")
st.caption("E4: LOL 욕설 + 특수문자 정상 데이터 증강으로 일반 욕설 탐지 강화 & 특수문자 오탐 감소 목표")

with st.sidebar:
    st.subheader("설정")
    threshold = st.slider("판정 임계값 (abusive)", 0.10, 0.90, 0.50, 0.05)
    st.write(f"Device: `{DEVICE}`")
    st.write(f"Max length: `{MAX_LEN}`")

# 모델 로드
try:
    tokenizer, model = load_artifacts()
except Exception as e:
    st.error("모델 로딩 실패")
    st.code(str(e))
    st.stop()

st.subheader("입력")
examples = [
    "ㅅㅂ",
    "시*발 뭐하냐",
    "진짜 개빡치네",
    "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
    "@@@",
    "좋은 하루 보내세요",
]
cols = st.columns(3)
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, use_container_width=True):
        st.session_state["text"] = ex

text = st.text_area("문장을 입력하세요", key="text", height=120, placeholder="예) ㅆㅣㅂㅏㄹ ㅋㅋ")

run = st.button("분석하기", type="primary", use_container_width=True)

if run:
    if not text.strip():
        st.warning("텍스트를 입력해줘!")
    else:
        p = predict_proba_abusive(text, tokenizer, model)
        pred = 1 if p >= threshold else 0

        st.subheader("결과")
        if pred == 1:
            st.error(f"🚨 {LABEL_MAP[pred]}")
        else:
            st.success(f"✅ {LABEL_MAP[pred]}")

        st.metric("악성 확률 p(abusive)", f"{p*100:.1f}%")
        st.progress(min(max(p, 0.0), 1.0))

        with st.expander("자세히 보기"):
            st.write(f"- 임계값: **{threshold:.2f}**")
            st.write(f"- p(abusive): **{p:.4f}**")
            st.write("- 참고: 모델은 오탐/미탐이 있을 수 있습니다.")
