import pandas as pd

# 엑셀 파일 경로
file_path = "/Users/jungjinho/Desktop/dataset.xlsx"

# 엑셀 파일 읽기
df = pd.read_excel(file_path)

# 'meta' 칼럼의 빈칸을 '논'으로 채우기
df['meta'] = df['meta'].fillna('0')

# 변경 사항을 원래 파일에 저장하기
df.to_excel(file_path, index=False)