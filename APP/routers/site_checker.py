from bs4 import BeautifulSoup
import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas.checker import SiteCheckRequest, SiteCheckResponse, RedirectData
from db.mongodb import MongoDB
from AI.ml_preprocessing import preprocess_and_vectorize
from AI.ml_models import train_and_evaluate
from AI.ml_storage import save_model, load_model
from utils.crawler import crawl_page

router = APIRouter()

mongo_instance = MongoDB()
collection = mongo_instance.get_collection("Search_Web", "User")
collection_html = mongo_instance.get_collection("Search_Web_HTML", "User")
ml_collection = mongo_instance.get_collection("Total_1", "Block")

ml_data = list(ml_collection.find({}))
df = pd.DataFrame(ml_data)
X_train, X_test, y_train, y_test, vectorizer = preprocess_and_vectorize(df)

stack_model, accuracy_stack = train_and_evaluate(X_train, X_test, y_train, y_test)

model_path = '/Users/jungjinho/Desktop/Exit/example/APP/AI/ai2.joblib'
save_model(stack_model, model_path)

loaded_model = load_model(model_path)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'}

@router.post("/check_site", response_model=SiteCheckResponse)
async def check_site(site_check_request: SiteCheckRequest):
    url = site_check_request.url

    if url == "chrome://newtab/" or "BlockedPage.html" in url or url == "http://127.0.0.1:5000/check":
        return SiteCheckResponse(url=url, blocked=False)


    collection.insert_one({"url": url})

    try:
        content = crawl_page(url, headers)
        soup = BeautifulSoup(content, 'html.parser')
        
        title = soup.title.text if soup.title else ""

        metas = soup.find_all("meta")
        meta_text = ' '.join([meta['content'] for meta in metas if meta.get('content')])

        a_texts = soup.find_all("a")
        a_text = ' '.join([a.text for a in a_texts])

        combined_text = title + " " + meta_text + " " + a_text
        combined_vector = vectorizer.transform([combined_text])
        
        collection_html.insert_one({
            "url": url,
            "title": title,
            "metas": [str(meta) for meta in metas],
            "a_text": [str(link) for link in a_texts]
        })
        
        prediction = loaded_model.predict(combined_vector)
        print(prediction) # [0], [1]
        
        blocked = bool(prediction[0])
        return SiteCheckResponse(url=url, blocked=blocked)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/save-redirect")
async def save_redirect(data: RedirectData):
    try:
        redirect_collection = mongo_instance.get_collection("status", "redirectUrl")
        redirect_collection.insert_one(data.dict())
        return {"message": "Data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
