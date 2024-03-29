from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from joblib import load
import requests
from bs4 import BeautifulSoup

# 저장된 TfidfVectorizer 모델 로드
vectorizer = joblib.load('/Users/jungjinho/Library/Mobile Documents/com~apple~CloudDocs/대학교/중부대학교/2024/2024 3-1/모의해킹:보안컨설팅프로젝트(캡스톤디자인)/프로그램/Visual Code/example/APP/AI/xgb_model.joblib')  # 저장된 TfidfVectorizer 모델의 경로로 수정해야 합니다.

# URL에서 데이터 추출 및 모델 예측
def predict_from_url(url):
    # URL의 HTTP Head 부분에서 title, meta, a 태그 추출
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.find('title').text if soup.find('title') else ""
    meta = ' '.join([meta.attrs.get('content', '') for meta in soup.find_all('meta')])
    a_text = ' '.join([a.text for a in soup.find_all('a')])
    
    combined_text = title + " " + meta + " " + a_text
    combined_vector = vectorizer.transform([combined_text])  # 벡터화
    
    # 저장된 모델 로드 및 예측
    loaded_model = joblib.load('/Users/jungjinho/Library/Mobile Documents/com~apple~CloudDocs/대학교/중부대학교/2024/2024 3-1/모의해킹:보안컨설팅프로젝트(캡스톤디자인)/프로그램/Visual Code/example/APP/AI/xgb_model.joblib')
    prediction = loaded_model.predict(combined_vector)
    
    print("예상 분류:", prediction)

# 예시 URL
predict_from_url("https://www.naver.com")
