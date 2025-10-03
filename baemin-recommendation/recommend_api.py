import os
import urllib

import certifi
from fastapi import FastAPI
from pymongo import MongoClient
from pymongo.server_api import ServerApi


MAPPING_EN2KO = {
    "hangover": "해장",
    "diet": "다이어트"
}
MAPPING_KO2EN = {v: k for k, v in MAPPING_EN2KO.items()}

app = FastAPI()

username = urllib.parse.quote_plus(os.environ['MONGODB_USERNAME'])
password = urllib.parse.quote_plus(os.environ['MONGODB_PASSWORD'])
uri = f"mongodb+srv://{username}:{password}@cluster0.61sar.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client.recommendations_db
collection = db.recommendations


@app.get("/health")
def health():
    """상태 확인용 엔드포인트."""
    return "OK"


@app.get("/recommend/{query_en}")
def recommend(query_en: str = "hangover"):
    """
    영어 키워드(예: hangover, diet)를 한글로 매핑해 MongoDB에서 추천 결과를 반환.
    - 입력: query_en (경로 파라미터)
    - 출력: 해당 카테고리의 추천 문서 목록
    """
    query_ko = MAPPING_EN2KO[query_en]
    data = list(collection.find({"_id": query_ko}, {'_id': 0}))
    return data