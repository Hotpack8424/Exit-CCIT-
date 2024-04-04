from fastapi import APIRouter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
import time

router = APIRouter()

def get_content_with_selenium(url):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    try:
        driver.get(url)
        time.sleep(5)
        page_content = driver.page_source
        return page_content
    finally:
        driver.quit()

def get_content_with_requests(url, headers):
    response = requests.get(url, headers=headers)
    return response.text

def crawl_page(url, headers):
    try:
        content = get_content_with_requests(url, headers)
    except Exception as e:
        print(f"requests를 사용한 {url} 접속 시도 중 에러: {e}, Selenium 시도 중...")
        content = get_content_with_selenium(url)
    return content
