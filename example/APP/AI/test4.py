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

# 파일 로드 및 데이터 정제
df = pd.read_csv('/Users/jungjinho/Library/Mobile Documents/com~apple~CloudDocs/대학교/중부대학교/2024/2024 3-1/모의해킹:보안컨설팅프로젝트(캡스톤디자인)/프로그램/데이터베이스_자료/분류_06/최종/mixing.csv')
df['url_len'] = pd.to_numeric(df['url_len'], errors='coerce')

df.dropna(subset=['url_len'], inplace=True)

Q1 = df['url_len'].quantile(0.25)
Q3 = df['url_len'].quantile(0.75)
IQR = Q3 - Q1
filter = (df['url_len'] >= Q1 - 1.5 * IQR) & (df['url_len'] <= Q3 + 1.5 * IQR)
df = df.loc[filter]

df['js_len'] = pd.to_numeric(df['js_len'], errors='coerce').fillna(0)

def get_geo_location(ip):
    try:
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url)
        data = response.json()
        return f"{data.get('country', '국가 정보 없음')}, {data.get('city', '도시 정보 없음')}"
    except Exception as e:
        return f"위치 정보 조회 오류: {str(e)}"

df['ip_geo'] = df['ip_add'].apply(get_geo_location)

df['ip_geo'] = df['ip_geo'].apply(lambda x: re.sub(r'[^가-힣A-Za-z0-9\s]', '', str(x)))

df.drop(['ip_add'], axis=1, inplace=True)

df['a'] = df['a'].str.replace(r"[^a-zA-Z\s]", "", regex=True)

def tokenize_and_remove_punctuation(text):
    tokens = word_tokenize(str(text))
    tokens = [re.sub(r"[^\w\s]", '', token) for token in tokens]
    tokens = [token for token in tokens if token]
    return tokens

df['a_Tokenized'] = df['a'].apply(tokenize_and_remove_punctuation)

df.drop('a', axis=1, inplace=True)

df.dropna(inplace=True)

df['a_Tokenized'] = df['a_Tokenized'].apply(lambda x: [word for word in x if word.startswith('http')])

df['a_Tokenized'] = df['a_Tokenized'].apply(lambda x: ' '.join(x))

df['a_url_count'] = df['a_Tokenized'].apply(lambda x: len(x.split()))

df.drop(['a_Tokenized'], axis=1, inplace=True)

# 텍스트 전처리 함수
def preprocess_text(text):
    if pd.isna(text):
        return np.nan
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', text).strip()
    return '\n'.join(kss.split_sentences(text))

# 문장 분리 및 전처리 적용
df['processed_meta'] = df['meta'].apply(preprocess_text)

# 명사 추출 함수
def extract_nouns(texts):
    komoran = Komoran()
    stopwords = ['이', '그', '저', '것', '수', '등', '때']
    return [' '.join([noun for noun in komoran.nouns(text) if noun not in stopwords and len(noun) > 1]) for text in texts if not pd.isna(text)]

# 명사 추출
df['nouns'] = extract_nouns(df['processed_meta'])

# 'processed_meta', 'meta' 칼럼 및 다른 불필요한 칼럼 제거
df.drop(['processed_meta', 'meta', 'connect', 'title', 'https','url'], axis=1, inplace=True)

# 'nouns' 칼럼의 이름을 'meta'로 변경
df.rename(columns={'nouns': 'meta'}, inplace=True)

df[['country', 'city']] = df['ip_geo'].str.split(' ', n=1, expand=True)

# 각 범주형 변수마다 새로운 LabelEncoder 인스턴스 사용
city_encoder = LabelEncoder()
country_encoder = LabelEncoder()
tld_encoder = LabelEncoder()

df['city_encoded'] = city_encoder.fit_transform(df['city'].astype(str))
df['country_encoded'] = country_encoder.fit_transform(df['country'].astype(str))
df['tld_encoded'] = tld_encoder.fit_transform(df['tld'].astype(str))

df['country'] = df['country'].str.strip().str.lower()
df['city'] = df['city'].str.strip().str.lower()
df['country'] = df['country'].fillna('unknown')  # '0' 대신 'unknown' 사용
df['city'] = df['city'].fillna('unknown')  # '0' 대신 'unknown' 사용

df.drop(['ip_geo', 'tld', 'city', 'country'], axis=1, inplace=True)  # 원래 열 제거
# 'meta' 칼럼 단어 나눔
df['meta'] = df['meta'].apply(lambda x: ' '.join(str(x).split()))

# 'meta' 칼럼에서 전처리된 텍스트 데이터 추출
preprocessed_texts = df['meta'].tolist()

# TfidfVectorizer 인스턴스 생성 및 TF-IDF 행렬 계산
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Replace empty strings with NaN and then fill NaN with 0
df.replace('', np.nan, inplace=True)
df.fillna(0, inplace=True)

# Convert all columns to float
X_numeric = df.drop(['meta', 'label'], axis=1).astype(float)
# 레이블을 제외한 모든 열을 선택합니다.

# 모든 열을 float 타입으로 변환
X_numeric = X_numeric.astype(float)

# TF-IDF 행렬과 다른 열 결합
X_combined = hstack([X_numeric, tfidf_matrix])

# 타겟 변수 설정
y = df['label']  # 'label'을 실제 레이블 열 이름으로 교체해야 합니다.

# 데이터를 훈련 세트와 검증 세트로 나눔
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

print(f"훈련 세트 크기: {X_train.shape}, 검증 세트 크기: {X_val.shape}")

# XGBoost 모델 초기화
model = xgb.XGBClassifier()

# 모델 피팅
model.fit(X_train, y_train)

# 검증 세트에 대한 예측
y_pred = model.predict(X_val)

# 정확도 계산
accuracy = (y_pred == y_val).mean()
print(f"검증 세트 정확도: {accuracy:.2f}")