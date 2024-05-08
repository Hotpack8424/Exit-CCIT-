document.addEventListener('DOMContentLoaded', function() {
    var toggleSwitch = document.getElementById('toggleSwitch');

    chrome.storage.local.get(['isExtensionEnabled'], function(result) {
        toggleSwitch.checked = result.isExtensionEnabled !== undefined ? result.isExtensionEnabled : true;

        updateRedirectsDisplay(toggleSwitch.checked);
    });

    toggleSwitch.addEventListener('change', function() {
        chrome.storage.local.set({'isExtensionEnabled': toggleSwitch.checked});

        updateRedirectsDisplay(toggleSwitch.checked);

        chrome.runtime.sendMessage({ query: 'updateExtensionState', isEnabled: toggleSwitch.checked });
    });
});

function updateRedirectsDisplay(isEnabled) {
    if (isEnabled) {
        chrome.runtime.sendMessage({ query: 'getRedirects' }, (response) => {
            displayRedirects(response.redirects);
        });
    } else {
        displayRedirects([]);
    }
}

function displayRedirects(redirects) {
    const container = document.getElementById('redirects');
    container.innerHTML = '';
    if (redirects.length === 0) {
        container.textContent = '리다이렉션이 감지되지 않았습니다.';
    } else {
        const latestRedirect = redirects[redirects.length - 1];
        const element = document.createElement('p');
        element.textContent = `URL: ${latestRedirect.url} -> ${latestRedirect.redirectUrl} (상태 코드: ${latestRedirect.statusCode})`;
        container.appendChild(element);
    }
}
