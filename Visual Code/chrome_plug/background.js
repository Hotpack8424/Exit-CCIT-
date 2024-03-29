chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // URL이 로드 완료될 때만 실행
  if (changeInfo.status === 'complete' && tab.url) {
    // FastAPI 서버로 URL 전송
    fetch('http://127.0.0.1:8000/check_site', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({url: tab.url}),
    })
    .then(response => response.json()) // 서버로부터 받은 응답을 JSON으로 변환
    .then(data => {
      console.log(data)
      // 차단된 사이트일 경우
      if (data.blocked) {
        // 해당 탭을 닫음
        chrome.tabs.remove(tabId);
      }
    })
    .catch(error => console.error('Error:', error)); // 오류 처리
  }
});
