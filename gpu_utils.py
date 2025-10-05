"""
GPU 설정 및 유틸리티 함수
모든 프로젝트에서 공통으로 사용할 수 있는 GPU 관련 함수들
"""
import os
import torch


def get_device(prefer_gpu=True, device_id=0):
    """
    사용 가능한 디바이스를 반환합니다.
    
    Args:
        prefer_gpu (bool): GPU 사용을 선호하는지 여부 (기본값: True)
        device_id (int): 사용할 GPU ID (기본값: 0)
    
    Returns:
        torch.device: 사용할 디바이스 객체
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f'✓ GPU 사용: {torch.cuda.get_device_name(device_id)}')
        print(f'  - CUDA 버전: {torch.version.cuda}')
        print(f'  - GPU 메모리: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB')
    else:
        device = torch.device('cpu')
        if prefer_gpu:
            print('⚠ GPU를 사용할 수 없습니다. CPU를 사용합니다.')
        else:
            print('✓ CPU 사용')
    
    return device


def get_gpu_info():
    """
    GPU 정보를 출력합니다.
    
    Returns:
        dict: GPU 정보 딕셔너리
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__
    }
    
    if info['cuda_available']:
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(info['gpu_count'])]
        info['gpu_memory'] = [
            f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            for i in range(info['gpu_count'])
        ]
    
    return info


def print_gpu_info():
    """GPU 정보를 보기 좋게 출력합니다."""
    info = get_gpu_info()
    
    print("\n" + "="*50)
    print("GPU 정보")
    print("="*50)
    print(f"PyTorch 버전: {info['pytorch_version']}")
    print(f"CUDA 사용 가능: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA 버전: {info['cuda_version']}")
        print(f"GPU 개수: {info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(info['gpu_names'], info['gpu_memory'])):
            print(f"  GPU {i}: {name} ({memory})")
    else:
        print("GPU를 사용할 수 없습니다.")
    print("="*50 + "\n")


def set_seed(seed=42):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed (int): 시드 값 (기본값: 42)
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ 시드 설정 완료: {seed}")


def clear_gpu_memory():
    """GPU 메모리를 정리합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU 메모리 정리 완료")
    else:
        print("⚠ GPU를 사용할 수 없습니다.")


# 사용 예시
if __name__ == "__main__":
    # GPU 정보 출력
    print_gpu_info()
    
    # 디바이스 설정
    device = get_device(prefer_gpu=True)
    print(f"\n사용 중인 디바이스: {device}")
    
    # 시드 설정
    set_seed(42)
    
    # 간단한 텐서 연산 테스트
    print("\n" + "="*50)
    print("GPU 테스트")
    print("="*50)
    
    # 텐서 생성 및 GPU로 이동
    x = torch.randn(1000, 1000)
    x = x.to(device)
    
    # 간단한 연산
    import time
    start = time.time()
    y = torch.matmul(x, x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"행렬 곱셈 (1000x1000) 소요 시간: {elapsed*1000:.2f}ms")
    print(f"결과 텐서 디바이스: {y.device}")
    print("="*50 + "\n")
    
    # GPU 메모리 정리
    clear_gpu_memory()
