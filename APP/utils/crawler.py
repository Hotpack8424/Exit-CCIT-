#APP/utils/crawler.py
# 크롤링 로직
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import time


# Selenium을 사용하여 웹 페이지 크롤링
def get_content_with_selenium(url):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    try:
        driver.get(url)
        time.sleep(5)  # 페이지가 완전히 로드될 때까지 기다림
        page_content = driver.page_source
        return page_content
    finally:
        driver.quit()

# requests와 BeautifulSoup을 사용하여 웹 페이지 크롤링
def get_content_with_requests(url, headers):
    response = requests.get(url, headers=headers)
    return response.text
def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 여기에서 필요한 데이터 추출 로직을 구현합니다.
    return soup

# 웹 페이지 크롤링하여 내용 추출
def crawl_page(url, headers):
    try:
        content = get_content_with_requests(url, headers)
    except Exception as e:
        print(f"requests를 사용한 {url} 접속 시도 중 에러: {e}, Selenium 시도 중...")
        content = get_content_with_selenium(url)
    return content

