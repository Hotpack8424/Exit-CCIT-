from fastapi import APIRouter, HTTPException
from pymongo import MongoClient
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import time
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

router = APIRouter()

client = MongoClient("mongodb://localhost:27017")
db = client.Search_Web
collection = db.User

db_html = client.Search_Web_HTML
collection_html = db_html.User

ml_db = client.Total_1
ml_collection = ml_db.Block

# MongoDB에서 데이터 로드
ml_data = list(ml_collection.find({}))

# MongoDB 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(ml_data)

# MongoDB 데이터 전처리
df.fillna('', inplace=True)
X = df['url'] + " " + df['connect'] + " " + df['meta'] + " " + df['a'] + " " + df['title'] + " " + df['tld'] + " " + df['https'] + " " + df['ip']
y = df['label']

# 텍스트 데이터 벡터화
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 모델 정의 및 하이퍼파라미터 튜닝
rf = make_pipeline(StandardScaler(with_mean=False), RandomForestClassifier(n_estimators=200, random_state=42))
lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=1, random_state=42))
svm = make_pipeline(StandardScaler(with_mean=False), SVC(C=1, kernel='rbf', probability=True, random_state=42))
knn = make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=5))
xgb = make_pipeline(StandardScaler(with_mean=False), XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3, random_state=42))

base_models = [('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn), ('xgb', xgb)]
stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

# 스태킹 모델 학습
stack_model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_stack = stack_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)

# 모델 저장
dump(stack_model, '/Users/jungjinho/Desktop/ai2.joblib')
loaded_model = load('/Users/jungjinho/Desktop/ai2.joblib')

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'}

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
def get_content_with_requests(url):
    response = requests.get(url, headers=headers)
    return response.text

# 웹 페이지 크롤링하여 내용 추출
def crawl_page(url):
    try:
        content = get_content_with_requests(url)
    except Exception as e:
        print(f"requests를 사용한 {url} 접속 시도 중 에러: {e}, Selenium 시도 중...")
        content = get_content_with_selenium(url)
    return content

class SiteCheckRequest(BaseModel):
    url: str

class SiteCheckResponse(BaseModel):
    url: str
    blocked: bool

@router.post("/check_site", response_model=SiteCheckResponse)
async def check_site(site_check_request: SiteCheckRequest):
    url = site_check_request.url

    # URL이 chrome://newtab/인 경우 처리
    if url == "chrome://newtab/":
        return SiteCheckResponse(url=url, blocked=False)

    # URL을 MongoDB에 저장
    collection.insert_one({"url": url})
    
    try:
        content = crawl_page(url)
        soup = BeautifulSoup(content, 'html.parser')
        
        title = soup.title.text if soup.title else ""

        metas = soup.find_all("meta")
        meta_text = ' '.join([meta['content'] for meta in metas if meta.get('content')])

        a_texts = soup.find_all("a")
        a_text = ' '.join([a.text for a in a_texts])

        combined_text = title + " " + meta_text + " " + a_text
        combined_vector = vectorizer.transform([combined_text])
        
        # 웹 크롤링 결과를 MongoDB에 저장
        collection_html.insert_one({
            "url": url,
            "title": title,
            "metas": meta_text,
            "a_text": a_text
        })
        
        prediction = loaded_model.predict(combined_vector)
        print(prediction) # [0], [1]
        
        blocked = bool(prediction[0])
        return SiteCheckResponse(url=url, blocked=blocked)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
