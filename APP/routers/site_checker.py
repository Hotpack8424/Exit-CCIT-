from email import header
from fastapi import APIRouter, HTTPException
from ..schemas import SiteCheckRequest, SiteCheckResponse
from ..db.mongodb import get_collection

from ..AI.ml_preprocessing import preprocess_and_vectorize
from ..AI.ml_models import train_and_evaluate
from ..AI.ml_storage import save_model, load_model
from ..utils.crawler import get_content_with_requests, get_content_with_selenium, crawl_page, parse_html

router = APIRouter()

# db/mongodb.py 모듈에서 정의된 함수 사용
collection = get_collection("User")
collection_html = get_collection("User", db_name="Search_Web_HTML")
ml_collection = get_collection("Block", db_name="Total_1")

# MongoDB에서 데이터 로드 및 전처리
ml_data = list(ml_collection.find({}))
df = pd.DataFrame(ml_data)
X_train, X_test, y_train, y_test, vectorizer = preprocess_and_vectorize(df)

# 모델 학습 및 평가
stack_model, accuracy_stack = train_and_evaluate(X_train, X_test, y_train, y_test)

# 모델 저장
model_path = 'APP/AI/saved_models/ai_model.joblib'
save_model(stack_model, model_path)

# 저장된 모델 로딩
loaded_model = load_model(model_path)





# APP/routers/site_checker.py
from bs4 import BeautifulSoup
from pydantic import BaseModel
import pandas as pd
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'
    }


class SiteCheckRequest(BaseModel):
    url: str

class SiteCheckResponse(BaseModel):
    url: str
    blocked: bool

@router.post("/check_site", response_model=SiteCheckResponse)
async def check_site(site_check_request: SiteCheckRequest):
    url = site_check_request.url

    # URL이 chrome://newtab/인 경우 처리
    if url == "chrome://newtab/":
        return SiteCheckResponse(url=url, blocked=False)

    # URL을 MongoDB에 저장
    collection.insert_one({"url": url})
    
    try:
        content = crawl_page(url)
        soup = BeautifulSoup(content, 'html.parser')
        
        title = soup.title.text if soup.title else ""

        metas = soup.find_all("meta")
        meta_text = ' '.join([meta['content'] for meta in metas if meta.get('content')])

        a_texts = soup.find_all("a")
        a_text = ' '.join([a.text for a in a_texts])

        combined_text = title + " " + meta_text + " " + a_text
        combined_vector = vectorizer.transform([combined_text])
        
        # 웹 크롤링 결과를 MongoDB에 저장
        collection_html.insert_one({
            "url": url,
            "title": title,
            "metas": meta_text,
            "a_text": a_text
        })
        
        prediction = loaded_model.predict(combined_vector)
        print(prediction) # [0], [1]
        
        blocked = bool(prediction[0])
        return SiteCheckResponse(url=url, blocked=blocked)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))