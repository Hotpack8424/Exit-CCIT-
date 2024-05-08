from flask import Flask, request, render_template_string
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'}

def check_url_with_selenium(url):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    try:
        driver.get(url)
        time.sleep(5)
        print(f"{url} 접속 가능 (Selenium을 통한 접속)")
        return True
    except Exception as e:
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
        response = requests.get(url, headers=headers, allow_redirects=False)
        if response.status_code in [301, 302, 303, 307, 308]:
            location = response.headers.get('Location')
            if location:
                print(f"{url} 리다이렉션 발생: {location}")
                check_url(location, depth=depth+1, max_depth=max_depth, results=results, retry_delay=retry_delay)
            else:
                print(f"{url} 리다이렉션 위치를 찾을 수 없음")
                results['blocked'] += 1
        elif response.status_code >= 200 and response.status_code < 300:
            print(f"{url} 접속 가능")
            results['accessible'] += 1
        elif response.status_code == 403:
            if check_url_with_selenium(url):
                results['accessible'] += 1
            else:
                results['blocked'] += 1
        else:
            print(f"{url} 접속 불가능 / 상태 코드: {response.status_code}")
            results['blocked'] += 1
    except requests.exceptions.RequestException as e:
        print(f"{url} 접속 불가능. 에러: {e}")
        if "Connection aborted" in str(e) or "dh key too small" in str(e) or "Max retries exceeded" in str(e) or "nodename nor servname provided, or not known" in str(e):
            if check_url_with_selenium(url):
                results['accessible'] += 1
            else:
                results['blocked'] += 1
        else:
            results['blocked'] += 1

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>유해 사이트 판별 사이트</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Noto Sans KR', sans-serif;
            text-align: center;
            background-color: #f0f2f5;
            margin: 0;
            padding: 40px 20px;
            color: #333;
        }

        header {
            margin-bottom: 40px;
        }

        h1 {
            color: #212529;
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 0.5em;
        }

        h2 {
            color: #6c757d;
            font-size: 1.25rem;
            font-weight: 400;
            margin-bottom: 1.5em;
        }

        input[type="text"] {
            padding: 12px 20px;
            font-size: 1rem;
            border: 2px solid #ced4da;
            border-radius: 30px;
            width: 300px;
            outline: none;
            color: #495057;
            transition: border-color 0.3s ease-in-out;
        }

        input[type="text"]:focus {
            border-color: #80bdff;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
        }

        button {
            padding: 12px 24px;
            font-size: 1rem;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            outline: none;
        }

        button:hover {
            background-color: #0056b3;
        }

        .warning {
            color: #dc3545;
            font-weight: bold;
            margin-top: 20px;
        }

        footer {
            margin-top: 60px;
            color: #6c757d;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <header>
        <a href="/check" style="position: absolute; left: 20px; top: 20px; font-size: 1rem; color: #007bff; text-decoration: none;">홈으로</a>
        <h1>당신이 접속할 사이트가 안전한가요?</h1>
        <h2>도메인 주소를 한 번 입력 해보세요.</h2>
    </header>
    
    <section>
        {% if message %}
            <p>{{ message }}</p>
        {% endif %}
        <form action="/check" method="post">
            <input type="text" name="url" placeholder="검색어를 입력하세요..." class="search-input">
            <button type="submit">검색</button>
        </form>
    </section>

    <p class="warning">[ 주의 사항 ] URL 입력 시, http://, https://를 포함해 입력해주세요.</p>

    <footer>
        <p>Copyright © 2024 All rights reserved by 비상구.</p>
    </footer>
</body>
</html>
"""

@app.route('/check', methods=['GET', 'POST'])
def check_url_route():
    message = None
    if request.method == 'POST':
        url = request.form['url']
        results = {'accessible': 0, 'blocked': 0}
        check_url(url, results=results)
        if results['accessible'] > 0:
            message = f"{url} 브라우저는 접속이 가능합니다."
        else:
            message = f"{url} 브라우저는 차단되어 접속이 불가능합니다."
    return render_template_string(HTML_TEMPLATE, message=message)

if __name__ == '__main__':
    app.run(debug=True)
