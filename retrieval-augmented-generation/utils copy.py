import os
from pathlib import Path
from openai import OpenAI


def load_api_key(env_file_name='env.txt'):
    """
    env.txt 파일에서 OPENAI_API_KEY를 로드합니다.
    
    Args:
        env_file_name (str): 환경변수 파일 이름 (기본값: 'env.txt')
    
    Returns:
        str: OPENAI_API_KEY 값
    
    Raises:
        ValueError: API 키를 찾을 수 없는 경우
    """
    # 현재 파일의 디렉토리를 기준으로 env 파일 경로 설정
    current_dir = Path(__file__).parent
    env_file = current_dir / env_file_name
    
    # 파일 내용을 읽어서 환경변수 설정
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # API 키 가져오기
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY를 찾을 수 없습니다. env.txt 파일을 확인해주세요.")
    
    return api_key


def get_openai_client(env_file_name='env.txt'):
    """
    OpenAI 클라이언트를 생성하여 반환합니다.
    
    Args:
        env_file_name (str): 환경변수 파일 이름 (기본값: 'env.txt')
    
    Returns:
        OpenAI: OpenAI 클라이언트 객체
    """
    api_key = load_api_key(env_file_name)
    return OpenAI(api_key=api_key)

