import os
from pathlib import Path
from openai import OpenAI


def load_opapi_key():
    current_dir = Path(__file__).parent
    env_file = current_dir / 'env.txt'
    
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY를 찾을 수 없습니다. env.txt 파일을 확인해주세요.")
    
    return api_key

def get_openai_client():
    api_key = load_opapi_key()
    return api_key, OpenAI(api_key=api_key) 