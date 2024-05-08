from fastapi import APIRouter
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

router = APIRouter()

def train_and_evaluate(X_train, X_val, y_train, y_val):
    # XGBoost 모델 생성
    xgb_model = XGBClassifier()

    # 하이퍼파라미터 그리드 정의
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.1],
        'n_estimators': [450, 500, 550],
        'max_depth': [3, 4, 5]
    }

    # 그리드 서치 객체 생성
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

    # 그리드 서치를 사용하여 모델 훈련
    grid_search.fit(X_train, y_train)

    # 최적의 모델 저장
    best_xgb_model = grid_search.best_estimator_

    # 최적의 모델로 전체 훈련 데이터에 학습
    best_xgb_model.fit(X_train, y_train)

    # 검증 세트를 사용하여 최적의 모델 평가
    val_accuracy = best_xgb_model.score(X_val, y_val)

    # 최적의 XGBoost 모델과 검증 세트의 정확도 반환
    return best_xgb_model, val_accuracy
