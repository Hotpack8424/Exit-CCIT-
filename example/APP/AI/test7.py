import pandas as pd
import numpy as np
import re
import kss
import requests
from konlpy.tag import Komoran
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report
from pymongo import MongoClient

# 저장된 모델과 컴포넌트 로딩
model_path = '/Users/jungjinho/Desktop/Exit/example/APP/AI/best_xgb_model04.joblib'
loaded_model, loaded_vectorizer, loaded_encoders = joblib.load(model_path)

client = MongoClient('mongodb://localhost:27017/')
db = client['Search_Web_HTML']  # 데이터베이스 이름
collection = db['User']  # 컬렉션 이름

# 새 데이터 로드
documents = collection.find({}, {"_id": 0})  # 모든 문서를 조회, 필요에 따라 쿼리 조건을 추가할 수 있음
df = pd.DataFrame(list(documents))


df['url_len'] = pd.to_numeric(df['url_len'], errors='coerce')
df.dropna(subset=['url_len'], inplace=True)
filter = (df['url_len'] >= df['url_len'].quantile(0.25) - 1.5 * (df['url_len'].quantile(0.75) - df['url_len'].quantile(0.25))) & (df['url_len'] <= df['url_len'].quantile(0.75) + 1.5 * (df['url_len'].quantile(0.75) - df['url_len'].quantile(0.25)))
df = df.loc[filter]
df['js_len'] = pd.to_numeric(df['js_len'], errors='coerce').fillna(0)

# 지리적 위치 정보 가져오기
def get_geo_location(ip):
    try:
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url)
        data = response.json()
        return f"{data.get('country', 'Unknown')}, {data.get('city', 'Unknown')}"
    except Exception as e:
        return f"위치 정보 조회 오류: {str(e)}"

df['ip_geo'] = df['ip_add'].apply(get_geo_location)
df['ip_geo'] = df['ip_geo'].apply(lambda x: re.sub(r'[^가-힣A-Za-z0-9\s]', '', str(x)))
df.drop(['ip_add'], axis=1, inplace=True)

df['a'] = df['a'].str.replace(r"[^a-zA-Z\s]", "", regex=True)

def tokenize_and_remove_punctuation(text):
    tokens = word_tokenize(str(text))
    tokens = [re.sub(r"[^\w\s]", '', token) for token in tokens]
    return [token for token in tokens if token]

df['a_Tokenized'] = df['a'].apply(tokenize_and_remove_punctuation)
df.drop('a', axis=1, inplace=True)
df.dropna(inplace=True)
df['a_Tokenized'] = df['a_Tokenized'].apply(lambda x: ' '.join([word for word in x if word.startswith('http')]))
df['a_url_count'] = df['a_Tokenized'].apply(lambda x: len(x.split()))
df.drop(['a_Tokenized'], axis=1, inplace=True)

def preprocess_text(text):
    if pd.isna(text):
        return np.nan
    return '\n'.join(kss.split_sentences(re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', text).strip()))

df['processed_meta'] = df['meta'].apply(preprocess_text)

def extract_nouns(texts):
    komoran = Komoran()
    stopwords = {'이', '그', '저', '것', '수', '등', '때'}
    return [' '.join([noun for noun in komoran.nouns(text) if noun not in stopwords and len(noun) > 1]) for text in texts if not pd.isna(text)]

df['nouns'] = extract_nouns(df['processed_meta'])
df.drop(['processed_meta', 'meta'], axis=1, inplace=True)
df.rename(columns={'nouns': 'meta'}, inplace=True)

# 'ip_geo' 열을 나라와 도시로 분할 전 데이터 확인 및 처리
df['ip_geo'] = df['ip_geo'].fillna('Unknown, Unknown')  # NaN 값에 대한 기본값 설정
df['ip_geo'] = df['ip_geo'].apply(lambda x: x if ', ' in x else x + ', Unknown')  # 쉼표가 없는 경우 기본 도시 값 추가

# 이제 안전하게 나눌 수 있습니다.
df[['country', 'city']] = df['ip_geo'].str.split(', ', expand=True)

def safe_transform(encoder, series):
    # 시리즈에서 NaN을 'Unknown'으로 대체하고 소문자로 변환
    series = series.fillna('Unknown').str.lower()
    # 'Unknown'이 encoder의 classes_에 포함되어 있는지 확인하고 포함되지 않았다면 추가
    if 'Unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'Unknown')
    return encoder.transform(series)

def clean_labels(text):
    text = str(text).lower().strip()
    if "unknown" in text:
        return "unknown"
    # 'south korea goyangsi'와 같은 특이한 레이블을 일반적인 범주로 매핑합니다.
    if "goyangsi" in text:
        return "south korea"
    return text

df['country'] = df['country'].apply(clean_labels)
df['city'] = df['city'].apply(clean_labels)
df['tld'] = df['tld'].apply(clean_labels)

df['country'] = 'unknown'

print(df['country'].unique())
print(df['city'].unique())
print(df['tld'].unique())



# 레이블 인코딩 적용
for col in ['country', 'city', 'tld']:
    encoder = loaded_encoders[col]
    # 'Unknown' 클래스가 없는 경우 추가
    if 'Unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'Unknown')

# 범주형 열의 NaN 값 처리
df[['country', 'city', 'tld']] = df[['country', 'city', 'tld']].fillna('Unknown')

# 보이지 않는 라벨을 처리하는 라벨 인코딩 적용
for col in ['country', 'city', 'tld']:
    encoder = loaded_encoders[col]
    df[col + '_encoded'] = safe_transform(encoder, df[col])


# 나머지 데이터 처리 및 TF-IDF 벡터화, 모델 예측 등의 과정을 계속 진행합니다.


df['meta'] = df['meta'].astype(str)  # 'meta' 열을 문자열로 강제 변환
tfidf_features = loaded_vectorizer.transform(df['meta'])  # TF-IDF 벡터화


df.drop(['ip_geo', 'tld', 'city', 'country', 'meta'], axis=1, inplace=True)
numeric_features = df.select_dtypes(include=[np.number])
X = hstack([numeric_features.values, tfidf_features])  # 수치형 특성과 TF-IDF 특성 결합


print(df.head(20))
predictions = loaded_model.predict(X)  # 예측 수행
results_df = pd.DataFrame({'Predicted Label': predictions})  # 결과 DataFrame 생성


print(predictions)
print(type(predictions))
print(results_df)  # 결과 출력
blocked = bool(predictions[0])
print(blocked)

# 인코딩 결과를 저장할 DataFrame 생성
encoded_results = pd.DataFrame({   
    'Encoded Country': df['country_encoded'],   
    'Encoded City': df['city_encoded'], 
    'Encoded TLD': df['tld_encoded']
})