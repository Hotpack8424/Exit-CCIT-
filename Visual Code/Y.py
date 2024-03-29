import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# 엑셀 파일 읽기
urls_df = pd.read_excel('/Users/jungjinho/Downloads/Mixing.xlsx')  # 엑셀 파일 경로 수정

# Selenium 설정
s = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)

# 모델 및 벡터라이저 로드
loaded_model = joblib.load('/Users/jungjinho/Desktop/ai2.joblib')

vectorizer = TfidfVectorizer()

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
    
    prediction = loaded_model.predict(combined_vector)
    return prediction[0]  # 첫 번째 예측값 반환

# 예측 결과를 저장할 빈 리스트
predictions = []

# 각 URL에 대해 예측 수행
for url in urls_df['URL']:
    prediction = predict_from_url_with_selenium(url)
    predictions.append(prediction)

# 웹드라이버 종료
driver.quit()

# 예측 결과를 새 DataFrame에 저장
results_df = pd.DataFrame({
    'URL': urls_df['URL'],
    'Prediction': predictions
})

# 결과를 엑셀 파일로 저장

