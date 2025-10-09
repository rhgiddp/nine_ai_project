import os
from pathlib import Path
import pickle
from openai import OpenAI
from utils import get_openai_key

def get_eval_data():
    """평가용 대화 데이터 로드."""
    with open('./res/eval_data.pickle', 'rb') as f:
        eval_data = pickle.load(f)

    return eval_data


# question --> conversation, depth & creativity & level of detailedness 제외, instruction following 추가
def pointwise_eval(conversation, answer_a):
    """
    모델 응답을 1~10점으로 평가하는 프롬프트를 구성해 OpenAI로 점수/설명을 받습니다.
    반환: 평가 결과 텍스트(설명 + Rating:[[n]])
    """
    openai_api_key = get_openai_key()
    if not openai_api_key:
        raise ValueError("OpenAI API 키를 찾을 수 없습니다. env.txt 파일에 OPENAI_API_KEY를 설정해주세요.")
    
    client = OpenAI(api_key=openai_api_key)
    eval_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user conversation displayed below. Your evaluation should consider factors
such as the helpfulness, relevance, and accuracy.
Begin your evaluation by providing a short explanation.The response should be
between 1 to 5 sentences. Be as objective as
possible. After providing your explanation, please rate the response on a scale of 1 to 10
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
[User Conversation]
{conversation}
[The Start of Assistant’s Answer]
{answer_a}
[The End of Assistant’s Answer]"""
    
    completion = client.chat.completions.create(
        model='gpt-4o-2024-05-13',
        messages=[{'role': 'user', 'content': eval_prompt}],
        temperature=0.0
    )

    return completion.choices[0].message.content