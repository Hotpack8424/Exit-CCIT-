import pandas as pd
import requests
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive'
}

def check_url(url, depth=0, max_depth=10, results={'accessible': 0, 'blocked': 0}, retry_delay=200, max_attempts=3):
    if depth > max_depth:
        print(f"{url} 접속 불가능: 최대 리다이렉션 깊이 초과")
        results['blocked'] += 1
        return

    attempts = 0  # 재시도 횟수 카운트
    while attempts < max_attempts:
        try:
            response = requests.get(url, allow_redirects=True)
            if response.status_code >= 200 and response.status_code < 300:
                print(f"{url} 접속 가능")
                results['accessible'] += 1
                return  # 성공 시 함수 종료
            elif 400 <= response.status_code < 500:
                print(f"{url} 재시도 중... (시도 {attempts + 1}/{max_attempts})")
                attempts += 1
                time.sleep(retry_delay)  # 설정된 대기 시간(초)만큼 대기 후 재시도
            else:
                print(f"{url} 접속 불가능 / 상태 코드: {response.status_code}")
                results['blocked'] += 1
                return  # 400번대 외의 에러 코드는 재시도하지 않음
        except requests.exceptions.RequestException as e:
            print(f"{url} 접속 불가능. 에러: {e}")
            results['blocked'] += 1
            return  # 예외 발생 시 함수 종료

    # 모든 재시도가 실패한 경우
    print(f"{url} 최대 재시도 횟수 초과로 접속 불가능")
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

check_urls_from_excel("/Users/jungjinho/Library/Mobile Documents/com~apple~CloudDocs/대학교/중부대학교/2024/2024 3-1/모의해킹:보안컨설팅프로젝트(캡스톤디자인)/프로그램/데이터베이스_자료/Total.xlsx")
