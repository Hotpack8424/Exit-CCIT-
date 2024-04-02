document.addEventListener('DOMContentLoaded', function() {
    var toggleSwitch = document.getElementById('toggleSwitch');

    chrome.storage.local.get(['isExtensionEnabled'], function(result) {
        if (result.isExtensionEnabled === undefined) {
            chrome.storage.local.set({'isExtensionEnabled': true});
            toggleSwitch.checked = true;
        } else {
            toggleSwitch.checked = result.isExtensionEnabled;
        }
    });

    toggleSwitch.addEventListener('change', function() {
        chrome.storage.local.set({'isExtensionEnabled': toggleSwitch.checked});

        chrome.runtime.sendMessage({ query: 'updateExtensionState', isEnabled: toggleSwitch.checked });
    });
});

// 리다이렉션 출력 결과
function displayRedirects(redirects) {
    const container = document.getElementById('redirects');
    if (redirects.length === 0) {
        container.textContent = '리다이렉션이 감지되지 않았습니다.';
        return;
    }
    redirects.forEach((redirect) => {
        const element = document.createElement('p');
        element.textContent = `URL: ${redirect.url} -> ${redirect.redirectUrl} (상태 코드: ${redirect.statusCode})`;
        container.appendChild(element);
    });
}

chrome.runtime.sendMessage({ query: 'getRedirects' }, (response) => {
    displayRedirects(response.redirects);
});
