document.addEventListener('DOMContentLoaded', function() {
    var toggleSwitch = document.getElementById('toggleSwitch');

    chrome.storage.local.get(['isExtensionEnabled'], function(result) {
        toggleSwitch.checked = result.isExtensionEnabled !== undefined ? result.isExtensionEnabled : true;

        // 리다이렉션 정보 요청 및 표시 로직을 별도의 함수로 분리
        updateRedirectsDisplay(toggleSwitch.checked);
    });

    toggleSwitch.addEventListener('change', function() {
        chrome.storage.local.set({'isExtensionEnabled': toggleSwitch.checked});

        // 상태 변경 시 리다이렉션 정보 업데이트
        updateRedirectsDisplay(toggleSwitch.checked);

        chrome.runtime.sendMessage({ query: 'updateExtensionState', isEnabled: toggleSwitch.checked });
    });
});

function updateRedirectsDisplay(isEnabled) {
    if (isEnabled) {
        chrome.runtime.sendMessage({ query: 'getRedirects' }, (response) => {
            // 기존에 표시된 리다이렉션 정보를 비우고, 새로운 리다이렉션 정보만을 표시
            displayRedirects(response.redirects);
        });
    } else {
        displayRedirects([]); // 확장 프로그램이 비활성화된 경우, 빈 배열로 리다이렉션 정보 초기화
    }
}

function displayRedirects(redirects) {
    const container = document.getElementById('redirects');
    container.innerHTML = ''; // 기존에 표시된 내용을 초기화
    if (redirects.length === 0) {
        container.textContent = '리다이렉션이 감지되지 않았습니다.';
    } else {
        // 새로운 리다이렉션 정보만을 표시
        const latestRedirect = redirects[redirects.length - 1];
        const element = document.createElement('p');
        element.textContent = `URL: ${latestRedirect.url} -> ${latestRedirect.redirectUrl} (상태 코드: ${latestRedirect.statusCode})`;
        container.appendChild(element);
    }
}
