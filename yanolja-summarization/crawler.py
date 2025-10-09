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
    print(f"[시작] 크롤링 시작: {name}")
    print(f"[URL] {url}")
    
    review_list = []
    driver = None
    temp_profile_dir = None

    try:
        # Chrome 옵션 설정: 샌드박스/확장 비활성화, 임시 사용자 프로필 사용
        temp_profile_dir = tempfile.mkdtemp(prefix="yanolja_chrome_profile_")
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument(f"--user-data-dir={temp_profile_dir}")

        print("[브라우저] Chrome 브라우저 시작...")
        driver = webdriver.Chrome(options=chrome_options)
        
        print("[로딩] 페이지 로딩 중...")
        driver.get(url)
        time.sleep(3)

        # 무한 스크롤 대응: 일정 횟수만큼 스크롤 다운하여 리뷰 로드
        print("[스크롤] 리뷰 로딩을 위한 스크롤 시작...")
        scroll_count = 20
        for i in range(scroll_count):
            print(f"   스크롤 {i+1}/{scroll_count}")
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(2)

        print("[파싱] 페이지 파싱 중...")
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # 선택자 구조는 사이트 변경 시 깨질 수 있으므로 주기적 점검 필요
        review_containers = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div')
        review_date = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div > div.css-1toaz2b > div > div.css-1ivchjf')

        print(f"[발견] 발견된 리뷰 개수: {len(review_containers)}")

        for i in range(len(review_containers)):
            try:
                review_text = review_containers[i].find('p', class_='content-text').text
                review_stars = review_containers[i].select('path[fill="currentColor"]')
                star_cnt = sum(1 for star in review_stars if not star.has_attr('fill-rule'))
                date = review_date[i].text if i < len(review_date) else "날짜 없음"

                review_dict = {
                    'review': review_text,
                    'stars': star_cnt,
                    'date': date
                }

                review_list.append(review_dict)
            except Exception as e:
                print(f"[오류] 리뷰 {i+1} 파싱 오류: {e}")
                continue

        print(f"[저장] {len(review_list)}개 리뷰 저장 중...")
        with open(f'./res/{name}.json', 'w', encoding='utf-8') as f:
            json.dump(review_list, f, indent=4, ensure_ascii=False) 
            #ensure_ascii=False -> 아시아권은 false 처리
            #indent=4 -> 들여쓰기 4칸

        print(f"[완료] 크롤링 완료! {len(review_list)}개 리뷰가 ./res/{name}.json에 저장되었습니다.")

    except Exception as e:
        print(f"[오류] 크롤링 중 오류 발생: {e}")
        return False

    finally:
        # 종료 및 임시 프로필 정리
        if driver:
            try:
                driver.quit()
                print("[종료] 브라우저 종료")
            except:
                pass
        if temp_profile_dir:
            try:
                shutil.rmtree(temp_profile_dir, ignore_errors=True)
                print("[정리] 임시 파일 정리 완료")
            except:
                pass
    
    return True

# https://nol.yanolja.com/reviews/domestic/1000102261?sort=created-at%3Adesc -> reviews
# https://nol.yanolja.com/reviews/domestic/1000113873?sort=created-at%3Adesc -> ninetree_pangyo
# https://nol.yanolja.com/reviews/domestic/10048873?sort=created-at%3Adesc -> ninetree_yongsan
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("[오류] 사용법: python crawler.py <파일명> <URL>")
        print("예시: python crawler.py ninetree_pangyo \"https://nol.yanolja.com/reviews/domestic/1000113873?sort=created-at%3Adesc\"")
        sys.exit(1)
    
    name, url = sys.argv[1], sys.argv[2]
    
    # URL 검증
    if not url.startswith('http'):
        print("[오류] URL이 올바르지 않습니다. 'https://'로 시작해야 합니다.")
        sys.exit(1)
    
    success = crawl_yanolja_reviews(name=name, url=url)
    if success:
        print("[성공] 크롤링이 성공적으로 완료되었습니다!")
    else:
        print("[실패] 크롤링이 실패했습니다.")
        sys.exit(1)
