import os
from dotenv import load_dotenv

import numpy as np
from openai import OpenAI


# .env 로드 후 키 읽기
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError('환경변수 OPENAI_API_KEY가 없습니다. 프로젝트 폴더의 .env 또는 쉘에 설정하세요.')


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