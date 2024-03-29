from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from joblib import dump

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')  # MongoDB 서버 주소
db = client['Total']  # 'Total' 데이터베이스 선택
collection = db['Block']  # 'Block' 콜렉션 선택

# MongoDB에서 데이터 로드
data = list(collection.find({}))

# MongoDB 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(data)

# 입력 변수와 타겟 변수 설정
df.fillna('', inplace=True)
X = df['URL'] + " " + df['Type'] + " " + df['Name'] + " " + df['Content']
y = df['Content.1']

# 텍스트 데이터 벡터화
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 저장
dump(model, '/Users/jungjinho/Desktop/ai2.joblib')

# 예측 및 평가
predicted = model.predict(X_test)
print(classification_report(y_test, predicted))

# Selenium 설정
s = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)

# URL에서 데이터 추출 및 모델 예측 함수를 Selenium으로 변경
def predict_from_url_with_selenium(url):
    driver.get(url)
    driver.implicitly_wait(10)
    
    # Selenium을 사용하여 title, meta, a 태그의 텍스트 추출
    title = driver.find_element(By.TAG_NAME, "title").text if driver.find_elements(By.TAG_NAME, "title") else ""
    metas = driver.find_elements(By.TAG_NAME, "meta")
    meta_text = ' '.join([meta.get_attribute('content') for meta in metas if meta.get_attribute('content')])
    a_texts = driver.find_elements(By.TAG_NAME, "a")
    a_text = ' '.join([a.text for a in a_texts])
    
    combined_text = title + " " + meta_text + " " + a_text
    combined_vector = vectorizer.transform([combined_text])  # 벡터화
    
    # 저장된 모델 로드 및 예측
    loaded_model = joblib.load('/Users/jungjinho/Desktop/ai2.joblib')
    prediction = loaded_model.predict(combined_vector)
    
    print("예상 분류:", prediction)
    
# 함수 호출
predict_from_url_with_selenium("https://013ww.com")

# 웹드라이버 종료
driver.quit()