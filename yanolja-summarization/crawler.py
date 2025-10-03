import json
import sys
import time
import tempfile
import shutil

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def crawl_yanolja_reviews(name, url):
    """
    야놀자 숙소 상세 페이지에서 리뷰를 스크롤 로드한 뒤 파싱하여 저장합니다.

    - name: 결과 JSON 파일명 접두사 (예: "ninetree_yongsan")
    - url: 크롤링할 야놀자 숙소 상세 URL

    처리 흐름
      1) Chrome 옵션을 안전하게 구성(샌드박스/확장 비활성화, 임시 프로필)
      2) 페이지 끝까지 여러 번 스크롤하여 리뷰를 동적 로드
      3) BeautifulSoup으로 리뷰 텍스트/별점/날짜 파싱
      4) ./res/{name}.json 으로 저장
    """
    review_list = []

    # Chrome 옵션 설정: 샌드박스/확장 비활성화, 임시 사용자 프로필 사용
    temp_profile_dir = tempfile.mkdtemp(prefix="yanolja_chrome_profile_")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--user-data-dir={temp_profile_dir}")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    time.sleep(3)

    # 무한 스크롤 대응: 일정 횟수만큼 스크롤 다운하여 리뷰 로드
    scroll_count = 20
    for i in range(scroll_count):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(2)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 선택자 구조는 사이트 변경 시 깨질 수 있으므로 주기적 점검 필요
    review_containers = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div')
    review_date = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div > div.css-1toaz2b > div > div.css-1ivchjf')

    for i in range(len(review_containers)):
        review_text = review_containers[i].find('p', class_='content-text').text
        review_stars = review_containers[i].select('path[fill="currentColor"]')
        star_cnt = sum(1 for star in review_stars if not star.has_attr('fill-rule'))
        date = review_date[i].text

        review_dict = {
            'review': review_text,
            'stars': star_cnt,
            'date': date
        }

        review_list.append(review_dict)

    with open(f'./res/{name}.json', 'w') as f:
        json.dump(review_list, f, indent=4, ensure_ascii=False)

    # 종료 및 임시 프로필 정리
    try:
        driver.quit()
    finally:
        shutil.rmtree(temp_profile_dir, ignore_errors=True)


if __name__ == '__main__':
    name, url = sys.argv[1], sys.argv[2]
    crawl_yanolja_reviews(name=name, url=url)
