# 🎮 Jupyter Notebook에서 GPU 사용하는 방법

## ⚠️ 중요: 현재 상태

### 현재 프로젝트들은 GPU를 사용하지 않습니다!

**이유:**
- ✅ **OpenAI API** 사용 중 → 클라우드에서 실행 (로컬 GPU 불필요)
- ✅ **Anthropic API** 사용 중 → 클라우드에서 실행 (로컬 GPU 불필요)
- ✅ **Google AI API** 사용 중 → 클라우드에서 실행 (로컬 GPU 불필요)

**결론:**
- 현재 노트북들을 실행해도 **CPU만 사용**됩니다
- GPU는 **설치만 되어 있고 사용되지 않습니다**
- API 호출은 인터넷을 통해 클라우드 서버에서 처리됩니다

---

## 🚀 GPU를 사용하려면?

### 방법 1: Jupyter 커널 변경 (준비 완료!)

#### 1단계: Jupyter Notebook 실행
```bash
conda activate pytorch_gpu_312
jupyter notebook
```

#### 2단계: 노트북에서 커널 변경
1. Jupyter Notebook 열기
2. 상단 메뉴: **Kernel** → **Change kernel**
3. **"Python 3.12 (PyTorch GPU)"** 선택

#### 3단계: GPU 사용 코드 추가
```python
import torch

# GPU 확인
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# GPU에서 텐서 연산
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.matmul(x, y)
print(f"결과 텐서 디바이스: {z.device}")
```

---

### 방법 2: gpu_utils.py 사용 (더 간단!)

```python
# 노트북 첫 셀에 추가
import sys
sys.path.append('..')  # 프로젝트 루트로 경로 추가

from gpu_utils import get_device, print_gpu_info

# GPU 정보 출력
print_gpu_info()

# 디바이스 자동 선택 (GPU 있으면 GPU, 없으면 CPU)
device = get_device(prefer_gpu=True)

# 이제 PyTorch 코드에서 device 사용
import torch
x = torch.randn(100, 100, device=device)
```

---

## 📊 GPU가 실제로 필요한 경우

### 1. 로컬 LLM 실행
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device('cuda')

# 모델 로드 (GPU로)
model = AutoModelForCausalLM.from_pretrained(
    "beomi/Llama-3-Open-Ko-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")

# 추론
inputs = tokenizer("안녕하세요", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 2. 로컬 임베딩 모델
```python
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda')

# 임베딩 모델 로드 (GPU로)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = model.to(device)

# 임베딩 생성
sentences = ["안녕하세요", "Hello", "こんにちは"]
embeddings = model.encode(sentences, device=device)
print(embeddings.shape)
```

### 3. 파인튜닝/학습
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('cuda')

# 모델을 GPU로
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

# 학습 루프
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)  # 데이터를 GPU로
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🔍 GPU 사용 확인 방법

### 방법 1: 코드로 확인
```python
import torch

print("=" * 50)
print("GPU 사용 확인")
print("=" * 50)
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # 현재 GPU 메모리 사용량
    print(f"할당된 메모리: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    print(f"캐시된 메모리: {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB")
else:
    print("GPU를 사용할 수 없습니다.")
print("=" * 50)
```

### 방법 2: NVIDIA 도구 사용
```bash
# GPU 사용률 실시간 모니터링
nvidia-smi -l 1

# 또는 한 번만 확인
nvidia-smi
```

---

## 📝 현재 프로젝트별 상태

### 1. `assistant-question-answering/`
- **API 사용**: OpenAI API
- **GPU 필요**: ❌ 없음
- **현재 상태**: CPU만 사용 (정상)

### 2. `baemin-recommendation/`
- **API 사용**: OpenAI API
- **GPU 필요**: ❌ 없음
- **현재 상태**: CPU만 사용 (정상)

### 3. `kakaotalk-summarization/`
- **API 사용**: OpenAI, Anthropic, Google AI API
- **GPU 필요**: ❌ 없음
- **현재 상태**: CPU만 사용 (정상)

### 4. `prompt-engineering/`
- **API 사용**: OpenAI API
- **GPU 필요**: ❌ 없음
- **현재 상태**: CPU만 사용 (정상)

### 5. `retrieval-augmented-generation/`
- **API 사용**: OpenAI API (Embeddings)
- **GPU 필요**: ❌ 없음
- **현재 상태**: CPU만 사용 (정상)
- **참고**: 로컬 임베딩 모델 사용 시 GPU 필요

### 6. `yanolja-summarization/`
- **API 사용**: OpenAI API
- **GPU 필요**: ❌ 없음
- **현재 상태**: CPU만 사용 (정상)

---

## 💡 요약

### 현재 상황:
- ✅ GPU 환경 설정 완료 (`pytorch_gpu_312`)
- ✅ PyTorch + CUDA 설치 완료
- ✅ Jupyter 커널 등록 완료
- ❌ **하지만 현재 프로젝트들은 GPU를 사용하지 않음**

### 이유:
- 모든 프로젝트가 **클라우드 API** 사용
- API는 **원격 서버**에서 실행
- 로컬 GPU는 **필요 없음**

### GPU를 사용하려면:
1. **로컬 모델 사용** (Llama, Mistral 등)
2. **로컬 임베딩 모델 사용** (sentence-transformers 등)
3. **모델 파인튜닝/학습**
4. **대량 텐서 연산**

### 커널 변경 방법:
1. Jupyter Notebook 실행: `conda activate pytorch_gpu_312 && jupyter notebook`
2. 커널 변경: **Kernel** → **Change kernel** → **"Python 3.12 (PyTorch GPU)"**
3. GPU 코드 작성 (위 예제 참조)

---

**작성일**: 2025-10-06  
**GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)  
**환경**: `pytorch_gpu_312` (Python 3.12.11, PyTorch 2.5.1+cu121)
