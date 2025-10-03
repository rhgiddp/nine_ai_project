import os

import numpy as np
from openai import OpenAI


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


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