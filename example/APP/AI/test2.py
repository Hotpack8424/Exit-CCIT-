import pandas as pd
import numpy as np
import requests
import re
import time
import pandas as pd
import nltk
import kss
from konlpy.tag import Komoran
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

df = pd.read_excel('/Users/jungjinho/Downloads/mixing1.xlsx')


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

df.drop(['a_Tokenized', 'connect', 'title', 'https'], axis=1, inplace=True)

EMAIL_PATTERN = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
URL_PATTERN = re.compile(r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
NON_ALPHANUMERIC_PATTERN = re.compile(r'[^\w\s]')
NEWLINE_PATTERN = re.compile(r'\n')

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

def tfidf_transform_and_save(input_df, text_column, output_file_path):

    tfidf_vectorizer = TfidfVectorizer()

    cleaned_data = input_df[text_column].dropna()
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_data)

    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    tfidf_matrix_shape = tfidf_matrix.shape

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_features)

    tfidf_df.to_excel(output_file_path, index=False)

    print(f"TF-IDF 행렬 크기: {tfidf_matrix_shape}")
    print(f"결과가 저장된 파일 경로: {output_file_path}")

def main_preprocess_and_tfidf(df, text_column, output_preprocess_path, output_frequency_path, output_tfidf_path):

    apply_preprocessing_and_noun_extraction(df, output_preprocess_path, text_column, output_frequency_path)
    
    processed_data = pd.read_excel(output_preprocess_path)

    tfidf_transform_and_save(processed_data, text_column, output_tfidf_path)

output_preprocess_path = '/Users/jungjinho/Desktop/dataset.xlsx'
text_column = 'meta'
output_frequency_path = '/Users/jungjinho/Desktop/dataset333.txt'
output_tfidf_path = '/Users/jungjinho/Desktop/dataset222.xlsx'
main_preprocess_and_tfidf(df, text_column, output_preprocess_path, output_frequency_path, output_tfidf_path)