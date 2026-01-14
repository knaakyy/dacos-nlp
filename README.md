# 🛡️ 변형 욕설 탐지 프로젝트
### 25-2-team3 자연어처리 1팀

> 딥러닝 기반 한국어 변형 욕설 탐지 모델

기존 키워드 매칭 방식의 한계를 극복하고, **초성 욕설**, **특수문자 삽입**, **띄어쓰기 변형** 등 다양한 변형 욕설을 탐지하는 프로젝트입니다.

---

## 📋 목차

1. [프로젝트 소개](#1-프로젝트-소개)
2. [KcELECTRA](#2-kcelectra)
3. [FastText](#3-fasttext)
4. [파일 구조](#4-파일-구조)

---

## 1. 프로젝트 소개

### 🎯 목표
- 변형 욕설 탐지 성능 향상
- 기존 키워드 매칭 방식 대비 성능 비교 및 검증

### 🔍 변형 욕설이란?
| 유형 | 예시 |
|------|------|
| 초성 욕설 | ㅅㅂ, ㅂㅅ, ㅈㄹ |
| 숫자 삽입 | 시1발, ㅈ1랄 |
| 특수문자 삽입 | 씨@발, 쌈@뽕 |
| 띄어쓰기 변형 | ㅂ ㅅ, 시 발 |
| 모음 분리 | ㄱㅐㅅㄲ, 시ㅃㅏㄹ |

### 📊 사용 모델
| 모델 | 설명 | 상태 |
|------|------|------|
| **KcELECTRA** | 한국어 사전학습 언어모델 기반 | ✅ 완료 |
| **FastText** | 단어 임베딩 기반 분류 모델 | 🔜 예정 |

---

## 2. KcELECTRA

한국어 사전학습 언어모델 **KcELECTRA**를 파인튜닝하여 악성 댓글을 분류합니다.

### 2.1 설치 방법

#### 환경 요구사항
- Python 3.8+
- CUDA 11.8+ (GPU 사용 시)
- Anaconda 또는 Miniconda 권장

#### Conda 환경 생성

```bash
# 환경 생성
conda create -n kcelectra python=3.10 -y
conda activate kcelectra
```

#### 패키지 설치

```bash
# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기타 패키지
pip install transformers==4.42.0 datasets accelerate scikit-learn pandas matplotlib seaborn
```

> 💡 노트북의 첫 번째 셀을 실행하면 자동으로 설치됩니다.

### 2.2 실행 방법

#### 모델 학습

```bash
# 1) Conda 환경 활성화
conda activate kcelectra

# 2) Jupyter 노트북 실행
jupyter notebook kcelectra/KcElectra_학습.ipynb
```

**실행 순서:**
1. 패키지 설치 (최초 1회)
2. 경로 설정 (데이터 폴더 확인)
3. 셀 순서대로 실행

**출력물:**
- `kcelectra/kcelectra_output/best_model/` - 학습된 모델

#### 키워드 매칭 vs KcELECTRA 비교

```bash
jupyter notebook kcelectra/KcElectra_비교.ipynb
```

**⚠️ 주의:** 반드시 `KcElectra_학습.ipynb`를 먼저 실행하여 모델을 저장해야 합니다.

**출력물:**
- 성능 비교표 (Accuracy, Precision, Recall, F1)
- 카테고리별 탐지율 비교 차트
- 개별 예측 결과

### 2.3 성능 결과

| 카테고리 | 키워드 매칭 | KcELECTRA | 차이 |
|----------|------------|-----------|------|
| 일반 욕설 | 100% | 100% | - |
| **변형 욕설** | **20%** | **93%** | **+73%** |
| 정상 | 100% | 80% | -20% |

> 💡 변형 욕설 탐지에서 KcELECTRA가 압도적으로 높은 성능을 보입니다.

### 2.4 참고 자료

- [KcELECTRA (beomi/KcELECTRA-base-v2022)](https://huggingface.co/beomi/KcELECTRA-base-v2022)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 3. FastText

> 🔜 **작성 예정**

### 3.1 설치 방법

<!-- TODO: FastText 설치 방법 작성 -->

### 3.2 실행 방법

<!-- TODO: FastText 실행 방법 작성 -->

### 3.3 성능 결과

<!-- TODO: FastText 성능 결과 작성 -->

### 3.4 참고 자료

<!-- TODO: FastText 참고 자료 작성 -->

---

## 4. 파일 구조

```
25-2-team3/
│
├── README.md                    # 프로젝트 설명서
│
├── kcelectra/                   # KcELECTRA 모델
│   ├── KcElectra_학습.ipynb     # 모델 학습 노트북
│   ├── KcElectra_비교.ipynb     # 키워드 vs KcELECTRA 비교
│   └── kcelectra_output/        # 학습된 모델 저장 폴더 (자동 생성)
│       └── best_model/
│
└── fasttext/                    # FastText 모델 (예정)
    └── ...
```

---

## 👥 팀원

- DACOS 2025 2학기 프로젝트 Team 3
