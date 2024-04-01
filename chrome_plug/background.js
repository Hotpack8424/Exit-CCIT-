chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  chrome.storage.local.get(['isExtensionEnabled'], function(result) {
    if (result.isExtensionEnabled !== false) { // 기본값을 true로 설정
      if (changeInfo.status === 'complete' && tab.url) {
        fetch('http://127.0.0.1:8000/check_site', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({url: tab.url}),
        })
        .then(response => response.json())
        .then(data => {
          if (data.blocked) {
            chrome.tabs.update(tabId, {url: "BlockedPage.html"});
          }
        })
        .catch(error => console.error('Error:', error));
      }
    }
  });
});
