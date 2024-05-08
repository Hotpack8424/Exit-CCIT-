from bs4 import BeautifulSoup
import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas.checker import SiteCheckRequest, SiteCheckResponse, RedirectData
from db.mongodb import MongoDB
from AI.ml_preprocessing import preprocess_and_vectorize
# from AI.ml_models import train_and_evaluate
# from AI.ml_storage import save_model, load_model
from utils.crawler import crawl_page
from urllib.parse import urlparse
import re
import tldextract
import joblib
import socket
import requests
import whois
import datetime
import time

router = APIRouter()

mongo_instance = MongoDB()
collection = mongo_instance.get_collection("Search_Web", "User") # 사용자가 입력한 주소를 저장하는 데이터베이스
collection_html = mongo_instance.get_collection("Search_Web_HTML", "User") # 사용자가 입력한 주소의 웹 크롤링 정보를 저장하는 데이터베이스
ml_collection = mongo_instance.get_collection("Total_1", "Block") # 불법 유해 사이트에 대한 데이터베이스
harmful_collection = mongo_instance.get_collection("Total", "harmful") # 불법 유해 사이트의 URL을 모아 놓은 데이터베이스
notharmful_collection = mongo_instance.get_collection("Total", "notharmful") # 정상 사이트의 URL을 모아 놓은 데이터베이스

def is_official_tld(url):
    excel_file_path = '/Users/jungjinho/Desktop/Exit/example/APP/routers/Tld.xlsx'
    
    df = pd.read_excel(excel_file_path)
    official_domains = df['DOMAIN'].tolist()
    
    extracted = tldextract.extract(url)
    tld = extracted.suffix
    
    if tld in official_domains:
        return False
    else:
        return True

def check_and_block_similar_site(url, harmful_collection):
    domain_match = re.search(r'https?://([a-zA-Z]+)(\d+)(\.com)', url)
    if not domain_match:
        return False
    
    domain_name = domain_match.group(1)
    tld = domain_match.group(3)
    domain_pattern = f'{domain_name}[0-9]+{tld}'

    regex_pattern = re.compile(domain_pattern)
    similar_domains_count = harmful_collection.count_documents({"url": {"$regex": regex_pattern}})

    if similar_domains_count > 0:
        return True
    else:
        return False

def check_url(url, headers, depth=0, max_depth=10):
    if depth > max_depth:
        return False
    try:
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        if response.status_code >= 200 and response.status_code < 300:
            return True
        else:
            return False
    except Exception as e:
        return False

def extract_domain_name(url):
    extracted = tldextract.extract(url)
    domain_name = "{}.{}".format(extracted.domain, extracted.suffix)
    return domain_name

def get_domain_dates(url):
    try:
        domain = urlparse(url).netloc
        if domain:
            w = whois.whois(domain)
            # 날짜를 datetime 객체로 통일
            creation_date = normalize_date(w.creation_date)
            expiration_date = normalize_date(w.expiration_date)
            return creation_date, expiration_date
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching WHOIS for {url}: {e}")
        return None, None

def normalize_date(date):
    if isinstance(date, datetime.datetime):
        return date
    elif isinstance(date, list):
        return date[0]
    elif date is None:
        return None
    else:
        raise TypeError(f"Unknown date type: {type(date)}")

def get_redirects(url):
    try:
        response = requests.get(url, allow_redirects=True)
        redirect_count = len(response.history)
        final_url = response.url
        return redirect_count, final_url
    except requests.RequestException as e:
        print(f"Error during request: {e}")
        return None, None

model_path = '/Users/jungjinho/Desktop/Exit/example/APP/AI/xgb_model_01.pkl'
loaded_model = joblib.load(model_path)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; ARM Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'}

@router.post("/check_site", response_model=SiteCheckResponse)
async def check_site(site_check_request: SiteCheckRequest):
    url = site_check_request.url
    
    # tld 변수를 수정하여 registered_domain을 저장하도록 합니다.
    extracted_result = tldextract.extract(url)
    domain_name = extract_domain_name(url)
    domain_pattern = re.escape(domain_name)
    # registered_domain을 안전하게 접근합니다.
    registered_domain = extracted_result.registered_domain if extracted_result.registered_domain else ''

    if url == "chrome://newtab/" or "BlockedPage.html" in url or url == "http://127.0.0.1:5000/check":
        return SiteCheckResponse(url=url, blocked=False)
    
    collection.insert_one({"url": url})
    
    domain_name = extract_domain_name(url)
    domain_pattern = re.escape(domain_name)
    
    if notharmful_collection.count_documents({"URL": {"$regex": domain_pattern, "$options": "i"}}) > 0:
        return SiteCheckResponse(url=url, blocked=False)

    if harmful_collection.count_documents({"url": {"$regex": domain_pattern, "$options": "i"}}) > 0:
        return SiteCheckResponse(url=url, blocked=True)
    
    if check_and_block_similar_site(url, harmful_collection):
        return SiteCheckResponse(url=url, blocked=True)
    
    if not is_official_tld(url):
        return SiteCheckResponse(url=url, blocked=True)

    try:
        content = crawl_page(url, headers)
        soup = BeautifulSoup(content, 'html.parser')
        
        start_time = time.time()
        
        is_accessible = check_url(url, headers)

        meta_tags = soup.find_all('meta')
        meta_text = ' '.join([meta['content'] for meta in meta_tags if meta.get('content')])
        
        a_tags = soup.find_all('a')
        a_text = ' '.join([a.text for a in a_tags])
        
        title_tag = soup.title.text if soup.title else ''
        
        url_len = len(url)
        
        tld = tldextract.extract(url).suffix
        
        http = 'https' if url.startswith('https') else 'http'
        
        js_len = sum(len(script.text) for script in soup.find_all('script') if script.text)
        
        ip_add = ''
        if registered_domain:
            try:
                ip_add = socket.gethostbyname(registered_domain)
            except socket.gaierror:
                ip_add = ''
        
        a_href = [tag.get('href') for tag in a_tags if tag.get('href')]
        count_external_links = sum(1 for link in a_href if urlparse(link).netloc and urlparse(link).netloc != registered_domain)
        
        domain_dates = get_domain_dates(url)
        
        div_tag_count = len(soup.find_all('div'))
        
        overlays = soup.find_all(".modal, .overlay, [role='dialog'], .popup")
        overlay_count = len(overlays)
        
        image_count = len(soup.find_all('img'))
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        redirect_count, final_url = get_redirects(url)
        
        collection_html.insert_one({
            "url": url,
            'connect': '성공' if is_accessible else '실패',
            "meta": meta_text,
            "a": a_text,
            "title": title_tag,
            "url_len": url_len,
            "js_len": js_len,
            "tld": tld,
            "https": http,
            "ip_add": ip_add,
            'count_external_links': count_external_links,
            'domain_creation': domain_dates[0],
            'domain_expiration': domain_dates[1],
            'div_count': div_tag_count,
            'popup_count': overlay_count,
            'image_count': image_count,
            'loading_time': loading_time,
            'redirect_count': redirect_count,
            'final_url': final_url
        })
        
        data = list(collection_html.find({"url": url}, {"_id": 0}))
        df = pd.DataFrame(data)
        
        fin = preprocess_and_vectorize(df)
        
        print(fin)
        
        return SiteCheckResponse(url=url, blocked=fin)
    except Exception as e:
        print(f"Error occurred during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail="Error occurred during prediction")

@router.post("/save-redirect")
async def save_redirect(data: RedirectData):
    try:
        redirect_collection = mongo_instance.get_collection("status", "redirectUrl")
        redirect_collection.insert_one(data.dict())
        return {"message": "Data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
