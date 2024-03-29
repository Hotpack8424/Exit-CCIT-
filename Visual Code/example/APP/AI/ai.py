from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# MongoDB 클라이언트 설정
client = MongoClient('mongodb://localhost:27017/')
db = client['Total']  # Total 데이터베이스 선택
collection = db['Block']  # Block 콜렉션 선택

# 데이터를 DataFrame으로 변환
data = pd.DataFrame(list(collection.find()))

# 'Type'을 라벨로, 'Content'를 특성으로 선택
X = data['Content'].astype(str)  # 'Content' 필드의 모든 값을 문자열로 변환
le = LabelEncoder()
y = le.fit_transform(data['Content.1'])  # 'Type' 필드를 숫자 라벨로 변환

# 텍스트 데이터를 수치 벡터로 변환하기 위한 TfidfVectorizer 설정
# RandomForestClassifier와 함께 Pipeline 구성
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
])

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline을 사용하여 데이터 전처리 및 모델 학습
pipeline.fit(X_train, y_train)

# 모델 평가
predictions = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# 모델 저장
joblib_file = "//Users/jungjinho/Library/Mobile Documents/com~apple~CloudDocs/대학교/중부대학교/2024/2024 3-1/모의해킹:보안컨설팅프로젝트(캡스톤디자인)/프로그램/Visual Code/example/APP/AI/ai2.joblib"
joblib.dump(pipeline, joblib_file)

import requests
from bs4 import BeautifulSoup

def get_web_content(url):
    """주어진 URL에서 웹 페이지의 내용을 가져옵니다."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        print(f"웹 페이지를 가져오는 중 오류가 발생했습니다: {e}")
        return None

def extract_head_tags(html):
    """HTML 문서에서 title, meta, a 태그의 내용을 추출합니다."""
    soup = BeautifulSoup(html, 'html.parser')
    head_elements = []
    
    # title 태그 추출
    title = soup.find('title')
    if title:
        head_elements.append(title.text)
    
    # meta 태그의 description 추출
    meta_description = soup.find('meta', attrs={'name': 'description'})
    if meta_description:
        head_elements.append(meta_description.get('content', ''))
    
    # a 태그의 텍스트 추출
    for a_tag in soup.find_all('a'):
        text = a_tag.text.strip()
        if text:
            head_elements.append(text)
    
    return ' '.join(head_elements)

# URL 입력
url = input("분석할 웹 페이지의 URL을 입력하세요: ")
html_content = get_web_content(url)
if html_content:
    head_content = extract_head_tags(html_content)
    # 모델에 입력하기 위해 리스트로 변환
    head_content_for_prediction = [head_content]
    # 예측 실행
    prediction = pipeline.predict(head_content_for_prediction)
    # 라벨 역변환을 통해 원래의 타입을 얻습니다.
    predicted_label = le.inverse_transform(prediction)[0]
    print(f"예측된 타입: {predicted_label}")
else:
    print("웹 페이지의 내용을 가져올 수 없습니다.")
