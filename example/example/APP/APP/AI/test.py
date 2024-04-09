import pandas as pd
import numpy as np
import requests
import re
import pandas as pd
import kss
import nltk
from konlpy.tag import Komoran
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# 엑셀 파일에서 데이터를 불러옵니다.
df = pd.read_csv('/Users/jungjinho/Desktop/output_total.csv')

# 'tld' 칼럼의 고유값들을 추출합니다.
unique_tlds = df['tld'].unique()
df.dropna(inplace=True)

# 고유값들에 대해 인덱스를 생성합니다.
word_index = {word: i for i, word in enumerate(unique_tlds)}

# 원-핫 인코딩을 수행합니다.
encoded_tlds = []
for word in df['tld']:
    lst = [0]*len(unique_tlds)
    lst[word_index[word]] = 1
    encoded_tlds.append(lst)

# 생성된 원-핫 인코딩 리스트를 원래 'tld' 칼럼에 대입합니다.
df['tld'] = encoded_tlds

# 'url_len' 칼럼을 수치형으로 변환합니다. 변환할 수 없는 값은 NaN으로 설정합니다.
df['url_len'] = pd.to_numeric(df['url_len'], errors='coerce')

# NaN 값이 포함된 행을 제거합니다.
df.dropna(subset=['url_len'], inplace=True)

# url_len 칼럼에서 이상치 제거
Q1 = df['url_len'].quantile(0.25)
Q3 = df['url_len'].quantile(0.75)
IQR = Q3 - Q1
filter = (df['url_len'] >= Q1 - 1.5 * IQR) & (df['url_len'] <= Q3 + 1.5 * IQR)
df = df.loc[filter]

# js_len 칼럼의 문자열을 0으로 변환
df['js_len'] = pd.to_numeric(df['js_len'], errors='coerce').fillna(0)

# IP 주소 지리적 값으로 변환 함수 정의
def get_geo_location(ip):
    try:
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url)
        data = response.json()
        return f"{data.get('country', '국가 정보 없음')}, {data.get('city', '도시 정보 없음')}"
    except Exception as e:
        return f"위치 정보 조회 오류: {str(e)}"

# IP 주소 칼럼에 대해 지리적 위치 정보 조회
df['지리적 위치'] = df['IP 주소'].apply(get_geo_location)

# '지리적 위치' 칼럼에서 특수 문자 제거
df['지리적 위치'] = df['지리적 위치'].apply(lambda x: re.sub(r'[^가-힣A-Za-z0-9\s]', '', str(x)))

# 'IP 주소' 칼럼 제거
df.drop(['IP 주소'], axis=1, inplace=True)

# 여기서부터 두 번째 코드의 처리 시작

# '지리적 위치' 칼럼의 모든 텍스트를 단어 단위로 분리하여 저장
words = []
for text in df['지리적 위치'].dropna().astype(str):
    words.extend(text.split())

# 유니크한 단어들만 추출하기 위해 set 자료형을 사용 후, 정렬
unique_words = sorted(set(words))

# 유니크한 단어들에 대해 정수 ID를 할당
word_to_id = {word: id for id, word in enumerate(unique_words, start=0)}

# 원-핫 인코딩 함수 정의
def one_hot_encode(text):
    tokens = text.split()
    encoded = [0] * len(unique_words)
    for token in tokens:
        if token in word_to_id:
            encoded[word_to_id[token]] = 1
    return encoded

# '지리적 위치' 칼럼에 원-핫 인코딩 적용 및 업데이트
df['지리적 위치'] = df['지리적 위치'].apply(one_hot_encode)

# 'A' 열에서 숫자, 특수문자, 한글 제거
df['a'] = df['a'].str.replace(r"[^a-zA-Z\s]", "", regex=True)


# 'a' 칼럼을 토큰화하고, 토큰화된 결과에서 특수문자를 제거하는 함수
def tokenize_and_remove_punctuation(text):
    tokens = word_tokenize(str(text))
    tokens = [re.sub(r"[^\w\s]", '', token) for token in tokens]  # 특수문자 제거
    tokens = [token for token in tokens if token]  # 빈 토큰 제거
    return tokens

df['a_Tokenized'] = df['a'].apply(tokenize_and_remove_punctuation)

# 'a' 칼럼 삭제
df.drop('a', axis=1, inplace=True)

# NaN 값이 포함된 행 제거
df.dropna(inplace=True)

# 'a_Tokenized' 칼럼에서 'http'로 시작하는 모든 단어를 리스트 형태로 저장 (이전에는 'https'를 제외했으나 이제 포함합니다.)
df['a_Tokenized'] = df['a_Tokenized'].apply(lambda x: [word for word in x if word.startswith('http')])

# 리스트를 공백으로 구분된 문자열로 변환
df['a_Tokenized'] = df['a_Tokenized'].apply(lambda x: ' '.join(x))

# 'a_Tokenized' 칼럼에서 각 행의 'http'로 시작하는 단어의 개수를 세어 'http_count'라는 새로운 칼럼에 저장
df['http_count'] = df['a_Tokenized'].apply(lambda x: len(x.split()))

# 'a_Tokenized', 'connect', 'title', 'https' 칼럼을 삭제
df.drop(['a_Tokenized', 'connect', 'title', 'https'], axis=1, inplace=True)

# 정규 표현식 전역 변수로 정의
EMAIL_PATTERN = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
URL_PATTERN = re.compile(r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
NON_ALPHANUMERIC_PATTERN = re.compile(r'[^\w\s]')
NEWLINE_PATTERN = re.compile(r'\n')

# 텍스트 클린징 함수
def clean(text):
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', text)
    return text.strip()

def clean_str(text):
    patterns = [
        (EMAIL_PATTERN, ' '),
        (URL_PATTERN, ' '),
        (HTML_TAG_PATTERN, ' '),
        (NON_ALPHANUMERIC_PATTERN, ' '),
        (NEWLINE_PATTERN, ' '),
    ]

    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    text = clean(text)
    return text

def preprocess_text(text):
    if pd.isna(text):
        return text
    text = clean_str(text)
    return text.strip()

def split_sentences_and_process(df, text_column):
    processed_texts = []
    for text in df[text_column]:
        if pd.isna(text):
            processed_texts.append(np.nan)
            continue
        preprocessed_text = preprocess_text(text)
        sentences = kss.split_sentences(preprocessed_text)
        processed_texts.append('\n'.join(sentences))
    df[text_column] = processed_texts
    return df

# Komoran을 이용하여 텍스트에서 명사를 추출하는 함수
def extract_nouns_with_komoran(texts):
    komoran = Komoran()
    stopwords = ['이', '그', '저', '것', '수', '등', '때']
    extracted_nouns = []
    for text in texts:
        if pd.isna(text) or isinstance(text, float):
            text = 'nan'
        if text == 'nan' or text.strip() == '':
            extracted_nouns.append('')
            continue
        nouns = komoran.nouns(str(text))
        filtered_nouns = [noun for noun in nouns if len(noun) > 1 and noun not in stopwords]
        extracted_nouns.append(' '.join(filtered_nouns))
    return extracted_nouns

# 명사의 출현 빈도를 txt 파일로 저장하는 함수
def save_nouns_frequency_to_txt(nouns_list, output_txt_path):
    all_nouns = ' '.join(nouns_list).split()
    nouns_frequency = Counter(all_nouns)
    sorted_nouns_frequency = sorted(nouns_frequency.items(), key=lambda x: x[1], reverse=True)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for noun, frequency in sorted_nouns_frequency:
            f.write(f'{noun} : {frequency}\n')

def apply_preprocessing_and_noun_extraction(df, output_file_path, text_column, output_frequency_path):
    df[text_column] = df[text_column].fillna("")
    
    df_processed = split_sentences_and_process(df, text_column)
    
    if text_column in df_processed.columns:
        extracted_nouns = extract_nouns_with_komoran(df_processed[text_column])
        df_processed[text_column] = extracted_nouns
        df_processed.to_excel(output_file_path, index=False)
        
        save_nouns_frequency_to_txt(extracted_nouns, output_frequency_path)
    else:
        print(f'Error: "{text_column}" column not found in the input file.')

# TF-IDF 변환
def tfidf_transform_and_save(input_df, text_column, output_file_path):
    # TF-IDF 변환을 위한 Vectorizer 초기화 (모든 단어를 사용)
    tfidf_vectorizer = TfidfVectorizer()

    # 'text_column' 컬럼에서 NaN 값을 제거하고 TF-IDF 변환 수행
    cleaned_data = input_df[text_column].dropna()
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_data)

    # 변환된 TF-IDF 행렬의 크기와 특성(단어) 이름 확인
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    tfidf_matrix_shape = tfidf_matrix.shape

    # 변환된 TF-IDF 행렬을 DataFrame으로 변환
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_features)

    # DataFrame을 엑셀 파일로 저장
    tfidf_df.to_excel(output_file_path, index=False)

    print(f"TF-IDF 행렬 크기: {tfidf_matrix_shape}")
    print(f"결과가 저장된 파일 경로: {output_file_path}")

# 전처리 및 명사 추출, 그리고 TF-IDF 변환을 적용하는 메인 함수
def main_preprocess_and_tfidf(df, text_column, output_preprocess_path, output_frequency_path, output_tfidf_path):
    # 전처리 및 명사 추출
    apply_preprocessing_and_noun_extraction(df, output_preprocess_path, text_column, output_frequency_path)
    
    # 전처리 및 명사 추출한 데이터 로드
    processed_data = pd.read_excel(output_preprocess_path)

    # TF-IDF 변환 및 결과 저장
    tfidf_transform_and_save(processed_data, text_column, output_tfidf_path)

# 실행 예시
output_preprocess_path = '/Users/jungjinho/Desktop/final_a_etc_meta.xlsx'  # 처리된 데이터를 저장할 출력 파일 경로
text_column = 'meta'  # 텍스트 데이터가 포함된 컬럼명
output_frequency_path = '/Users/jungjinho/Desktop/nouns_frequency_final.txt'  # 명사 빈도수를 저장할 txt 파일 경로
output_tfidf_path = '/Users/jungjinho/Desktop/final_tfidf_meta.xlsx'  # TF-IDF 결과를 저장할 엑셀 파일 경로

main_preprocess_and_tfidf(df, text_column, output_preprocess_path, output_frequency_path, output_tfidf_path)