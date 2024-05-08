from fastapi import APIRouter
import pandas as pd
import numpy as np
import re
import kss
import requests
from konlpy.tag import Komoran
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
import joblib

router = APIRouter()

def preprocess_and_vectorize(df):
    model_path = '/Users/jungjinho/Desktop/Exit/example/APP/AI/xgb_model_01.pkl'
    loaded_model, loaded_vectorizer, loaded_encoders = joblib.load(model_path)
    
    df['url_len'] = pd.to_numeric(df['url_len'], errors='coerce')
    df.dropna(subset=['url_len'], inplace=True)
    filter = (df['url_len'] >= df['url_len'].quantile(0.25) - 1.5 * (df['url_len'].quantile(0.75) - df['url_len'].quantile(0.25))) & (df['url_len'] <= df['url_len'].quantile(0.75) + 1.5 * (df['url_len'].quantile(0.75) - df['url_len'].quantile(0.25)))
    df = df.loc[filter]
    df['js_len'] = pd.to_numeric(df['js_len'], errors='coerce').fillna(0)

    # 'domain_creation' 과 'domain_expiration' 칼럼을 datetime 형태로 변경
    df['domain_creation'] = pd.to_datetime(df['domain_creation'], errors='coerce')
    df['domain_expiration'] = pd.to_datetime(df['domain_expiration'], errors='coerce')

    # 도메인 만료일 - 도메인 등록일 = 유효기간 칼럼 생성
    df['validity_period'] = (df['domain_expiration'] - df['domain_creation']).dt.days

    # 아래 칼럼들을 수치화 한 이후에 결측치는 0으로 반영
    columns_to_convert = ['redirect_count', 'loading_time', 'image_count', 'popup_count', 'div_count', 'count_external_links', 'validity_period']
    for column in columns_to_convert:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(0)

    # 아래 칼럼 버림
    df.drop(columns=['domain_creation', 'domain_expiration', 'final_url'], inplace=True)

    def get_geo_location(ip):
        try:
            url = f"http://ip-api.com/json/{ip}"
            response = requests.get(url)
            data = response.json()
            return f"{data.get('country', 'Unknown')}, {data.get('city', 'Unknown')}"
        except Exception:
            return ", "  # 예외 발생 시 비어 있는 문자열 반환

    df['ip_geo'] = df['ip_add'].apply(get_geo_location)

    df['ip_geo'] = df['ip_geo'].apply(lambda x: re.sub(r'[^가-힣A-Za-z0-9\s]', '', str(x)))

    df.drop(['ip_add'], axis=1, inplace=True)

    # 'ip_geo' 열이 비어 있는 경우 기본값 설정
    df['ip_geo'] = df['ip_geo'].fillna('Unknown Unknown')

    # 'ip_geo' 열을 공백으로 분할
    def parse_location(ip_geo):
        parts = ip_geo.rsplit(maxsplit=1)  # 마지막 공백에서 분할
        if len(parts) < 2:
            parts.insert(0, 'Unknown')  # 만약 분할된 부분이 하나뿐이라면 'country'를 'Unknown'으로 설정
        return parts

    # 안전 분할 함수 적용
    df[['country', 'city']] = df['ip_geo'].apply(parse_location).apply(pd.Series)

    df['country'] = df['country'].str.strip().str.lower()
    df['city'] = df['city'].str.strip().str.lower()
    
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

    def clean_labels(text, encoder):
        text = str(text).lower().strip()
        if "unknown" in text:
            return "unknown"
        # 레이블이 encoder의 classes_에 없는 경우에만 추가
        if text not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, text)
        return text

    country_encoder = loaded_encoders['country']
    city_encoder = loaded_encoders['city']
    tld_encoder = loaded_encoders['tld']
    
    df['country'] = df['country'].apply(lambda x: clean_labels(x, country_encoder))
    df['city'] = df['city'].apply(lambda x: clean_labels(x, city_encoder))
    df['tld'] = df['tld'].apply(lambda x: clean_labels(x, tld_encoder))

    # 레이블 인코딩 적용
    for col in ['country', 'city', 'tld']:
        encoder = loaded_encoders[col]
        df[col] = df[col].apply(lambda x: clean_labels(x, encoder))
        # 'Unknown' 레이블을 명시적으로 추가합니다.
        if 'unknown' not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, 'unknown')
        df[col + '_encoded'] = encoder.transform(df[col])

    # 나머지 데이터 처리 및 TF-IDF 벡터화, 모델 예측 등의 과정을 계속 진행합니다.
    df['meta'] = df['meta'].astype(str)  # 'meta' 열을 문자열로 강제 변환
    tfidf_features = loaded_vectorizer.transform(df['meta'])  # TF-IDF 벡터화

    df.drop(['ip_geo', 'tld', 'city', 'country'], axis=1, inplace=True)
    numeric_features = df.select_dtypes(include=[np.number])
    X = hstack([numeric_features.values, tfidf_features])  # 수치형 특성과 TF-IDF 특성 결합
    print(X)
    
    predictions = loaded_model.predict(X)
    predictions_test = loaded_model.predict_proba(X)
    
    print(predictions_test)
    print(predictions)
    
    blocked = bool(predictions[0])

    return blocked
