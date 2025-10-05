# Python 3.12 + PyTorch GPU 환경 설정 가이드

## 📋 요약

- **기본 Python**: 3.13.5 (PyTorch 미지원)
- **새 환경**: `pytorch_gpu_312` (Python 3.12.11)
- **PyTorch**: 2.5.1 + CUDA 12.1
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)

## 🔧 설치 방법

### 1단계: Conda 환경 생성
```bash
conda create -n pytorch_gpu_312 python=3.12 -y
```

### 2단계: 환경 활성화
```bash
conda activate pytorch_gpu_312
```

### 3단계: PyTorch 설치 (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4단계: 프로젝트 의존성 설치
```bash
pip install python-dotenv openai anthropic google-generativeai gradio numpy pandas
```

### 5단계: 설치 확인
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

예상 출력:
```
PyTorch: 2.5.1+cu121
CUDA: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 💻 사용 방법

### Jupyter Notebook 실행
```bash
conda activate pytorch_gpu_312
jupyter notebook
```

### Python 스크립트 실행
```bash
conda activate pytorch_gpu_312
python your_script.py
```

### GPU 테스트
```bash
conda activate pytorch_gpu_312
python gpu_utils.py
```

## ❓ 왜 Python 3.12인가?

### PyTorch 지원 버전:
- ✅ Python 3.8, 3.9, 3.10, 3.11, **3.12** - 모두 지원
- ❌ Python 3.13 - 아직 미지원 (2025년 초 현재)

### 기본 Python이 3.13인 이유:
- Anaconda 기본 환경이 Python 3.13.5
- PyTorch가 아직 3.13을 지원하지 않음
- 따라서 별도 환경(`pytorch_gpu_312`)을 생성

### Python 3.12 선택 이유:
- PyTorch 공식 지원
- 최신 Python 기능 사용 가능
- 안정성과 호환성 우수
- 3.11보다 성능 향상

## 🔄 환경 전환

### 기본 환경 (Python 3.13)
```bash
conda deactivate
python --version  # Python 3.13.5
```

### GPU 환경 (Python 3.12)
```bash
conda activate pytorch_gpu_312
python --version  # Python 3.12.11
```

## 📦 설치된 패키지

### 핵심 패키지:
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

### AI/ML 라이브러리:
- `openai==2.1.0`
- `anthropic==0.69.0`
- `google-generativeai==0.8.5`

### 유틸리티:
- `python-dotenv==1.1.1`
- `gradio==5.49.0`
- `numpy==2.1.2`
- `pandas==2.3.3`

## 🎯 프로젝트별 사용

모든 프로젝트의 `requirements.txt`에 PyTorch가 추가되었습니다:

```bash
# 프로젝트 디렉토리로 이동
cd assistant-question-answering

# 환경 활성화
conda activate pytorch_gpu_312

# 의존성 설치
pip install -r requirements.txt

# Jupyter 실행
jupyter notebook
```

## 🚨 주의사항

### 1. 환경 활성화 필수
PyTorch를 사용하려면 반드시 `pytorch_gpu_312` 환경을 활성화해야 합니다:
```bash
conda activate pytorch_gpu_312
```

### 2. OpenAI API는 GPU 불필요
- 현재 프로젝트는 주로 OpenAI API 사용
- OpenAI API는 클라우드에서 실행
- 로컬 GPU는 사용되지 않음

### 3. GPU가 필요한 경우
- 로컬 LLM 실행 (Llama, Mistral 등)
- 임베딩 모델 로컬 실행
- 파인튜닝/학습
- 대량 텐서 연산

### 4. 메모리 제한
- RTX 3060 Laptop: 6GB VRAM
- 작은 모델(7B 이하)만 실행 가능
- 큰 모델은 양자화 필요

## 🔍 문제 해결

### GPU가 인식되지 않는 경우
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# PyTorch CUDA 확인
conda activate pytorch_gpu_312
python -c "import torch; print(torch.cuda.is_available())"
```

### 환경이 보이지 않는 경우
```bash
# 모든 환경 확인
conda env list

# 환경 재생성
conda create -n pytorch_gpu_312 python=3.12 -y
```

### 패키지 충돌
```bash
# 환경 삭제 후 재생성
conda deactivate
conda remove -n pytorch_gpu_312 --all -y
conda create -n pytorch_gpu_312 python=3.12 -y
```

## 📚 추가 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [CUDA 설치 가이드](https://docs.nvidia.com/cuda/)
- [Conda 환경 관리](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

**작성일**: 2025-10-06
**Python**: 3.12.11
**PyTorch**: 2.5.1+cu121
**CUDA**: 12.1
