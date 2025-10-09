import os
from pathlib import Path
import pickle
import time

import anthropic
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI

MAX_LEN = 3000

# API 키 캐시
_api_keys_loaded = False
_api_keys = {
    'OPENAI_API_KEY': None,
    'GOOGLE_API_KEY': None,
    'ANTHROPIC_API_KEY': None
}

def _load_api_keys():
    """API 키를 한 번만 로드하고 캐시합니다."""
    global _api_keys_loaded
    if _api_keys_loaded:
        return  # 이미 로드됨
    
    # 현재 파일의 디렉토리를 기준으로 env 파일 경로 설정
    current_dir = Path(__file__).parent
    env_file = current_dir / 'env.txt'
    
    # 파일 내용을 읽어서 환경변수 설정
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # API 키 가져오기
    _api_keys['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    _api_keys['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY')
    _api_keys['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY')
    
    if not _api_keys['OPENAI_API_KEY'] and not _api_keys['GOOGLE_API_KEY'] and not _api_keys['ANTHROPIC_API_KEY']:
        raise ValueError("API 키를 찾을 수 없습니다. OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY 중 최소 하나는 env.txt 파일에 설정되어야 합니다.")
    
    _api_keys_loaded = True


def get_openai_key():    
    _load_api_keys()
    return _api_keys['OPENAI_API_KEY']

def get_googleai_key():    
    _load_api_keys()
    return _api_keys['GOOGLE_API_KEY']

def get_anthropicai_key():    
    _load_api_keys()
    return _api_keys['ANTHROPIC_API_KEY']


def shorten_conv(conversation):
    """대화가 너무 길면 뒤쪽만 남기도록 앞부분을 잘라 길이를 제한."""
    shortened_len = len(conversation)
    lst = conversation.split('\n')
    for i, l in enumerate(lst):
        utterance_len = len(l)
        shortened_len -= utterance_len
        if shortened_len <= MAX_LEN:
            break

    lst_shortened = lst[i+1:]
    conv_shortened = '\n'.join(lst_shortened)
    return conv_shortened


def summarize(conversation, prompt, temperature=0.0, model='gpt-3.5-turbo-0125'):
    """
    모델별로 요약을 수행합니다.
    - gpt*: OpenAI Chat Completions
    - gemini*: Google Generative AI
    - claude*: Anthropic Messages API
    """
    if len(conversation) > MAX_LEN:
        conversation = shorten_conv(conversation)

    prompt = prompt + '\n\n' + conversation

    if 'gpt' in model:
        openai_api_key = get_openai_key()
        client = OpenAI(api_key=openai_api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature
        )

        return completion.choices[0].message.content
    elif 'gemini' in model:
        google_api_key = get_googleai_key()
        genai.configure(api_key=google_api_key)
        client = genai.GenerativeModel(model)
        response = client.generate_content(
            contents=prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        time.sleep(1)

        return response.text
    elif 'claude' in model:
        anthropic_api_key = get_anthropicai_key()
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        message = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )

        return message.content[0].text


def get_train_data():
    """학습 예시 대화 데이터를 로드."""
    with open('./res/train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)

    return train_data


def get_prompt():
    """요약 지침 프롬프트를 구성하여 반환."""
    conv_train = get_train_data()[0]  # 첫 번째 데이터 사용

    prompt = f"""당신은 요약 전문가입니다. 사용자 대화들이 주어졌을 때 요약하는 것이 당신의 목표입니다. 대화를 요약할 때는 다음 단계를 따라주세요:

1. 대화 참여자 파악: 대화에 참여하는 사람들의 수와 관계를 파악합니다.
2. 주제 식별: 대화의 주요 주제와 부차적인 주제들을 식별합니다.
3. 핵심 내용 추출: 각 주제에 대한 중요한 정보나 의견을 추출합니다.
4. 감정과 태도 분석: 대화 참여자들의 감정이나 태도를 파악합니다.
5. 맥락 이해: 대화의 전반적인 맥락과 배경을 이해합니다.
6. 특이사항 기록: 대화 중 특별히 눈에 띄는 점이나 중요한 사건을 기록합니다.
7. 요약문 작성: 위의 단계에서 얻은 정보를 바탕으로 간결하고 명확한 요약문을 작성합니다.
각 단계를 수행한 후, 최종적으로 전체 대화를 200자 내외로 요약해주세요.

아래는 예시 대화와 예시 요약 과정 및 결과 입니다.

예시 대화:
{conv_train}

예시 요약 과정
1. 대화 참여자 파악: P01과 P02 두 명이 친구 관계로 보이며, 서로를 "언니", "오빠"라고 부르는 것으로 보아 나이 차이가 있는 친구 사이입니다.

2. 주제 식별: 연애 경험과 남자친구에 대한 이야기가 주요 주제입니다. 구체적으로는 현재 남자친구의 나이, 과거 연애 경험, 연상/연하 연애에 대한 의견 등을 다루고 있습니다.

3. 핵심 내용 추출: P01은 주로 오빠(연상)와 사귀었고, P02는 동갑과 사귀고 있으며 1년째 사귀고 있습니다. 둘 다 연하 연애는 부담스러워하며 자상한 오빠를 선호합니다.

4. 감정과 태도 분석: 가벼운 톤으로 친근하게 대화하며, 연애에 대한 솔직한 의견을 나누고 있습니다.

따라서 다음과 같이 요약할 수 있습니다:
두 친구가 연애 경험을 공유하며 현재 남자친구와의 관계, 과거 연애 스타일, 연상/연하 연애에 대한 선호도에 대해 가벼운 톤으로 대화하고 있습니다.

예시 요약 결과
두 친구가 연애 경험을 공유하며 현재 남자친구와의 관계, 과거 연애 스타일, 연상/연하 연애에 대한 선호도에 대해 가벼운 톤으로 대화하고 있습니다.
    
아래 사용자 대화에 대해 3문장 내로 요약해주세요:"""
    return prompt