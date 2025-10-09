#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import tempfile
import shutil

def check_selector_methods():
    """다양한 방법으로 셀렉터를 확인하는 함수"""
    
    # Chrome 옵션 설정
    temp_profile_dir = tempfile.mkdtemp(prefix="yanolja_check_")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument(f"--user-data-dir={temp_profile_dir}")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # 야놀자 리뷰 페이지로 이동
        url = "https://nol.yanolja.com/reviews/domestic/1000102261?sort=created-at%3Adesc"
        driver.get(url)
        time.sleep(3)
        
        print("=== 셀렉터 확인 방법들 ===")
        
        # 1. 현재 코드에서 사용하는 셀렉터
        selector1 = '#__next > section > div > div.css-1js0bc8 > div > div > div'
        try:
            elements1 = driver.find_elements(By.CSS_SELECTOR, selector1)
            print(f"1. 현재 셀렉터: {len(elements1)}개 요소 발견")
        except Exception as e:
            print(f"1. 현재 셀렉터 오류: {e}")
        
        # 2. 더 간단한 셀렉터 시도
        selector2 = '[class*="css-1js0bc8"]'
        try:
            elements2 = driver.find_elements(By.CSS_SELECTOR, selector2)
            print(f"2. 클래스 포함 셀렉터: {len(elements2)}개 요소 발견")
        except Exception as e:
            print(f"2. 클래스 포함 셀렉터 오류: {e}")
        
        # 3. 리뷰 텍스트가 포함된 요소 찾기
        selector3 = 'p[class*="content-text"]'
        try:
            elements3 = driver.find_elements(By.CSS_SELECTOR, selector3)
            print(f"3. 리뷰 텍스트 셀렉터: {len(elements3)}개 요소 발견")
        except Exception as e:
            print(f"3. 리뷰 텍스트 셀렉터 오류: {e}")
        
        # 4. XPath로 찾기
        xpath1 = "//p[contains(@class, 'content-text')]"
        try:
            elements4 = driver.find_elements(By.XPATH, xpath1)
            print(f"4. XPath 셀렉터: {len(elements4)}개 요소 발견")
        except Exception as e:
            print(f"4. XPath 셀렉터 오류: {e}")
            
        print("\n=== DevTools에서 셀렉터 얻는 방법 ===")
        print("1. F12 → Elements 탭")
        print("2. 화살표 아이콘 클릭 (Select an element)")
        print("3. 리뷰 영역 클릭")
        print("4. 하이라이트된 요소 우클릭")
        print("5. Copy → Copy selector 선택")
        print("6. 또는 Copy → Copy full XPath 선택")
        
    finally:
        driver.quit()
        shutil.rmtree(temp_profile_dir, ignore_errors=True)

if __name__ == "__main__":
    check_selector_methods()
