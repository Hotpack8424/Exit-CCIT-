import pandas as pd
from pymongo import MongoClient

# 1. 엑셀 파일 읽기
file_path = '/Users/jungjinho/Desktop/Total.xlsx'  # 엑셀 파일 경로를 여기에 입력하세요.
df = pd.read_excel(file_path, engine='openpyxl')

# 2. MongoDB 설정 및 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['Total_1']  # 데이터베이스 이름 지정
collection = db['Block']  # 컬렉션 이름 지정

# 3. 데이터 MongoDB에 저장
data_dict = df.to_dict("records")  # DataFrame을 문서 목록으로 변환
collection.insert_many(data_dict)  # 데이터 MongoDB에 저장

print("데이터가 MongoDB에 저장되었습니다.")