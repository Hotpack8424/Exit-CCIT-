let redirects = [];
let isExtensionEnabled = true;

function updateIcon() {
    let path;
    if (isExtensionEnabled) {
        path = {
            "16": "image/image16.png",
            "48": "image/image48.png",
            "64": "image/image64.png",
            "128": "image/image128.png"
        };
    } else {
        path = {
            "16": "image/image16_red.png",
            "48": "image/image48_red.png",
            "64": "image/image64_red.png",
            "128": "image/image128_red.png"
        };
    }
    chrome.action.setIcon({path: path});
}

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
                chrome.tabs.update(tabId, {url: "https://hit-ant.my.canva.site/blockpage"});
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
        updateIcon();
        if (!isExtensionEnabled) {
            redirects = [];
        }
    }
});

updateIcon();
