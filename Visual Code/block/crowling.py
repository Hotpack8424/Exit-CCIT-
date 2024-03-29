import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

# URL 정보를 추출하는 함수
def extract_info(url, driver):
    data = []
    try:
        driver.get(url)
        
        # 제목 추출
        try:
            title = driver.find_element(By.TAG_NAME, 'title').get_attribute('innerText').strip()
            data.append({'URL': url, 'Type': 'title', 'Name': 'title', 'Content': title})
        except Exception as e:
            print(f"Error extracting title from {url}: {e}")
        
        # 메타 태그 추출
        try:
            meta_tags = driver.find_elements(By.TAG_NAME, 'meta')
            for meta in meta_tags:
                name = meta.get_attribute('name') or meta.get_attribute('property')
                content = meta.get_attribute('content')
                if name and content:
                    data.append({'URL': url, 'Type': 'meta', 'Name': name, 'Content': content})
        except Exception as e:
            print(f"Error extracting meta tags from {url}: {e}")
        
        # 링크(a 태그) 추출
        try:
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                if href:
                    data.append({'URL': url, 'Type': 'a', 'Name': 'href', 'Content': href})
        except Exception as e:
            print(f"Error extracting links from {url}: {e}")
    
    except WebDriverException as e:
        print(f"WebDriverException occurred with {url}: {e}")

    return data


# Excel 파일에서 URL을 읽고 정보를 추출하여 새 Excel 파일에 저장하는 함수
def process_urls_from_excel(file_path, output_path):
    df = pd.read_excel(file_path)
    all_data = []
    
    # 셀레니움 드라이버 설정
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    for url in df['URL']:
        data = extract_info(url, driver)
        all_data.extend(data)
    
    driver.quit()  # 드라이버 종료
    
    # 데이터를 DataFrame으로 변환 후 엑셀 파일로 저장
    df_output = pd.DataFrame(all_data)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_output.to_excel(writer, index=False)

    print("Data has been saved to Excel successfully.")

# 실행 예시
file_path = '/Users/jungjinho/Desktop/해외자료2.xlsx'
output_path = '/Users/jungjinho/Desktop/해외자료_all.xlsx'
process_urls_from_excel(file_path, output_path)
