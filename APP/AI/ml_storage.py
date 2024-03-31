
# 모델 저장 및 로딩 모듈
from joblib import dump, load

def save_model(model, path):
    dump(model, path)

def load_model(path):
    return load(path)
