import pandas as pd
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'}

def check_url_with_selenium(url):
    # Selenium 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    try:
        driver.get(url)
        time.sleep(5)  # 페이지가 완전히 로드될 때까지 기다림
        print(f"{url} 접속 가능 (Selenium을 통한 접속)")
        return True
    except Exception as e:
        # 'net::ERR_CONNECTION_RESET' 에러가 발생한 경우, 이를 무시하고 접속 가능으로 간주
        if 'net::ERR_CONNECTION_RESET' in str(e):
            print(f"{url} 접속 가능 (Selenium을 통한 접속, net::ERR_CONNECTION_RESET 에러 무시)")
            return True
        else:
            print(f"{url} 접속 불가능. 에러: {e}")
            return False
    finally:
        driver.quit()

def check_url(url, depth=0, max_depth=10, results={'accessible': 0, 'blocked': 0}, retry_delay=20):
    if depth > max_depth:
        print(f"{url} 접속 불가능: 최대 리다이렉션 깊이 초과")
        results['blocked'] += 1
        return

    try:
        response = requests.get(url, headers=headers, allow_redirects=True)
        if response.status_code >= 200 and response.status_code < 300:
            print(f"{url} 접속 가능")
            results['accessible'] += 1
        elif response.status_code == 403:  # 403 상태 코드를 받은 경우, Selenium을 사용하여 접속 시도
            if check_url_with_selenium(url):
                results['accessible'] += 1
            else:
                results['blocked'] += 1
        else:
            print(f"{url} 접속 불가능 / 상태 코드: {response.status_code}")
            results['blocked'] += 1
    except requests.exceptions.RequestException as e:
        print(f"{url} 접속 불가능. 에러: {e}")
        # 아래 조건에 'Max retries exceeded', 'nodename nor servname provided, or not known' 에러 메시지들을 추가합니다.
        if "Connection aborted" in str(e) or "dh key too small" in str(e) or "Max retries exceeded" in str(e) or "nodename nor servname provided, or not known" in str(e):
            if check_url_with_selenium(url):
                results['accessible'] += 1
            else:
                results['blocked'] += 1
        else:
            results['blocked'] += 1


def check_urls_from_excel(file_path):
    results = {'accessible': 0, 'blocked': 0}
    df = pd.read_excel(file_path)
    for url in df['URL']:
        check_url(url, results=results)

    print(f"접속 가능한 사이트 수: {results['accessible']}, 차단된 사이트 수: {results['blocked']}")
    if results['accessible'] + results['blocked'] > 0:
        block_rate = (results['blocked'] / (results['accessible'] + results['blocked'])) * 100
        print(f"사이트 차단율: {block_rate:.2f}%")
    else:
        print("검사된 사이트가 없습니다.")

check_urls_from_excel("/Users/jungjinho/Library/Mobile Documents/com~apple~CloudDocs/대학교/중부대학교/2024/2024 3-1/모의해킹:보안컨설팅프로젝트(캡스톤디자인)/프로그램/데이터베이스_자료/Total.xlsx")
