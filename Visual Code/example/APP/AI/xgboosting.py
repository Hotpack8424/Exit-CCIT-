import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['Total_1']  # 'Total' 데이터베이스 선택
collection = db['Block']  # 'Block' 컬렉션 선택

# MongoDB에서 데이터 로드
data = list(collection.find({}))

# MongoDB 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(data)

# MongoDB 데이터 전처리
df.fillna('', inplace=True)
X = df['url'] + " " + df['connect'] + " " + df['meta'] + " " + df['a'] + " " + df['title'] + " " + df['tld'] + " " + df['https'] + " " + df['ip']
y = df['label']

# 텍스트 데이터 벡터화
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 모델 정의 및 하이퍼파라미터 튜닝
rf = make_pipeline(StandardScaler(with_mean=False), RandomForestClassifier(n_estimators=200, random_state=42))
lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=1, random_state=42))
svm = make_pipeline(StandardScaler(with_mean=False), SVC(C=1, kernel='rbf', probability=True, random_state=42))
knn = make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=5))
xgb = make_pipeline(StandardScaler(with_mean=False), XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3, random_state=42))

base_models = [('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn), ('xgb', xgb)]
stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

# 스태킹 모델 학습
stack_model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_stack = stack_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)

print(f"스태킹 모델의 정확도: {accuracy_stack:.4f}")

# 모델 저장
model_save_path = '/Users/jungjinho/Desktop/stack_model.joblib'
dump(stack_model, model_save_path)
print(f"모델이 '{model_save_path}' 경로에 저장되었습니다.")
