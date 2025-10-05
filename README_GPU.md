# GPU 사용 가이드

모든 프로젝트에서 GPU(PyTorch)를 사용할 수 있도록 설정되었습니다.

## 🎯 설정 완료 사항

### 1. PyTorch 환경 구성
- **환경 이름**: `pytorch_gpu_312`
- **Python 버전**: 3.12.11
- **PyTorch 버전**: 2.5.1 (CUDA 12.1 지원)
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)

### 2. 설치된 패키지
```bash
conda activate pytorch_gpu_312
```

주요 패키지:
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`
- `openai`, `anthropic`, `google-generativeai`
- `gradio`, `python-dotenv`

### 3. 모든 프로젝트 requirements.txt 업데이트
다음 프로젝트들의 `requirements.txt`에 PyTorch가 추가되었습니다:
- `assistant-question-answering/`
- `baemin-recommendation/`
- `kakaotalk-summarization/`
- `prompt-engineering/`
- `retrieval-augmented-generation/`
- `yanolja-summarization/`

## 📝 사용 방법

### 1. Conda 환경 활성화
```bash
conda activate pytorch_gpu_312
```

### 2. Jupyter 노트북에서 GPU 사용

#### 방법 1: gpu_utils.py 사용 (권장)
```python
import sys
sys.path.append('..')  # 프로젝트 루트 경로 추가

from gpu_utils import get_device, print_gpu_info

# GPU 정보 확인
print_gpu_info()

# 디바이스 설정
device = get_device(prefer_gpu=True)

# 모델/데이터를 GPU로 이동
# model = model.to(device)
# data = data.to(device)
```

#### 방법 2: 직접 설정
```python
import torch

# GPU 사용 가능 확인
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
print(f'GPU 이름: {torch.cuda.get_device_name(0)}')

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 디바이스: {device}')

# 텐서를 GPU로 이동
x = torch.randn(100, 100).to(device)
```

### 3. Python 스크립트에서 GPU 사용
```python
from gpu_utils import get_device, set_seed, clear_gpu_memory

# 시드 설정 (재현성)
set_seed(42)

# 디바이스 설정
device = get_device(prefer_gpu=True)

# 모델 학습/추론 후 메모리 정리
clear_gpu_memory()
```

## 🔧 GPU 유틸리티 함수 (gpu_utils.py)

### 주요 함수:
- `get_device(prefer_gpu=True, device_id=0)`: 사용할 디바이스 반환
- `get_gpu_info()`: GPU 정보 딕셔너리 반환
- `print_gpu_info()`: GPU 정보 출력
- `set_seed(seed=42)`: 재현성을 위한 시드 설정
- `clear_gpu_memory()`: GPU 메모리 정리

### 테스트:
```bash
conda activate pytorch_gpu_312
python gpu_utils.py
```

## 💡 주의사항

### 1. OpenAI API는 GPU 불필요
현재 프로젝트들은 주로 **OpenAI API**를 사용하므로 로컬 GPU가 필요하지 않습니다.
- OpenAI API는 클라우드에서 실행됨
- 로컬 GPU는 사용되지 않음

### 2. GPU가 필요한 경우
다음과 같은 경우에 GPU가 유용합니다:
- **로컬 LLM 실행** (Llama, Mistral 등)
- **임베딩 모델 로컬 실행**
- **파인튜닝/학습**
- **대량의 텐서 연산**

### 3. GPU 메모리 제한
- RTX 3060 Laptop GPU: **6GB VRAM**
- 작은 모델(7B 이하)만 실행 가능
- 큰 모델은 양자화(quantization) 필요

## 🚀 다음 단계

### 로컬 LLM 실행하기
```bash
conda activate pytorch_gpu_312
pip install transformers accelerate bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device('cuda')

# 작은 모델 로드 (예: GPT-2)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 텍스트 생성
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 임베딩 모델 로컬 실행
```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

sentences = ["This is a test sentence", "Another example"]
embeddings = model.encode(sentences)
print(embeddings.shape)
```

## 📊 성능 비교

### GPU vs CPU (1000x1000 행렬 곱셈)
- **GPU (RTX 3060)**: ~36ms
- **CPU**: ~100-200ms (약 3-5배 느림)

## 🔗 참고 자료
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [CUDA 설치 가이드](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## ❓ 문제 해결

### GPU가 인식되지 않는 경우
```bash
# CUDA 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족 에러
```python
# 배치 크기 줄이기
# 모델 양자화 사용
# GPU 메모리 정리
torch.cuda.empty_cache()
```

### Python 버전 문제
- PyTorch는 Python 3.8-3.12 지원
- Python 3.13은 아직 미지원 (2025년 초 현재)
- `pytorch_gpu_312` 환경은 Python 3.12.11 사용

---

**설정 완료일**: 2025-10-06
**GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
**CUDA**: 12.7 (Driver) / 12.1 (PyTorch)
