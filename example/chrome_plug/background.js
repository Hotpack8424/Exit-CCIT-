let redirects = [];
let isExtensionEnabled = true;

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url && isExtensionEnabled) {
        fetch('http://127.0.0.1:8000/check_site', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({url: tab.url}),
        })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            if (data.blocked) {
                chrome.tabs.update(tabId, {url: "BlockedPage.html"});
            }
        })
        .catch(error => console.error('Error:', error));
    }
});

function sendRedirectData(redirectData) {
  fetch('http://localhost:8000/save-redirect', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(redirectData),
  })
  .then(response => response.json())
  .then(data => console.log('Data sent successfully:', data))
  .catch((error) => {
      console.error('Error sending data:', error);
  });
}

// 리다이렉션 탐지 함수
chrome.webRequest.onBeforeRedirect.addListener(
  function (details) {
      const redirectObj = {
          url: details.url,
          redirectUrl: details.redirectUrl,
          statusCode: details.statusCode,
      };
      redirects.push(redirectObj);
      sendRedirectData(redirectObj);
  },
  { urls: ['<all_urls>'] },
  ['responseHeaders']
);


chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.query === 'getRedirects') {
        sendResponse({ redirects: redirects });
    } else if (request.query === 'updateExtensionState') {
        isExtensionEnabled = request.isEnabled;
        if (!isExtensionEnabled) {
            redirects = [];
        }
    }
});
