import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from openai import OpenAI

def load_api_key(env_file_name='env.txt'):
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


def get_embedding(text, model='text-embedding-3-small'):
    """단일 문장 임베딩을 반환."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def get_embeddings(text, model='text-embedding-3-small'):
    """복수 입력 임베딩 리스트 반환."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    output = []
    for i in range(len(response.data)):
        output.append(response.data[i].embedding)
    return output


def cosine_similarity(a, b):
    """코사인 유사도 계산."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def call_openai(prompt, temperature=0.0, model='gpt-3.5-turbo-0125'):
    """채팅 완성 API 호출 유틸."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature
    )

    return completion.choices[0].message.content


def retrieve_context(question):
    """가이드북 문단을 분리하고 임베딩 기반 유사도로 최상 문맥 반환."""
    with open('./res/guidebook_full.txt', 'r') as f:
        contexts = f.read().split('\n\n')

    question_embedding = get_embeddings([question], model='text-embedding-3-small')[0]
    context_embeddings = get_embeddings(contexts, model='text-embedding-3-small')

    similarities = [cosine_similarity(question_embedding, context_embedding) for context_embedding in context_embeddings]

    most_relevant_index = np.argmax(similarities)
    print(contexts[most_relevant_index])
    return contexts[most_relevant_index]