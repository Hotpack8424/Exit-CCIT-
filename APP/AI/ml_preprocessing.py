
# 데이터 전처리, 백터화 모듈
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_and_vectorize(df):
    # 데이터 전처리
    df.fillna('', inplace=True)
    X = df['url'] + " " + df['connect'] + " " + df['meta'] + " " + df['a'] + " " + df['title'] + " " + df['tld'] + " " + df['https'] + " " + df['ip']
    y = df['label']

    # 텍스트 데이터 벡터화
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer
