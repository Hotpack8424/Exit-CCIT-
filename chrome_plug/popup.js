document.addEventListener('DOMContentLoaded', function() {
    var toggleSwitch = document.getElementById('toggleSwitch');
  
    // 저장된 확장 프로그램 상태를 로드하고, 값이 없다면 기본적으로 ON 상태로 설정합니다.
    chrome.storage.local.get(['isExtensionEnabled'], function(result) {
      if (result.isExtensionEnabled === undefined) {
        // 'isExtensionEnabled' 값이 설정되어 있지 않으면 true로 설정합니다.
        chrome.storage.local.set({'isExtensionEnabled': true});
        toggleSwitch.checked = true;
      } else {
        // 저장된 값에 따라 스위치 상태를 설정합니다.
        toggleSwitch.checked = result.isExtensionEnabled;
      }
    });
  
    // 스위치 클릭 이벤트 리스너를 추가합니다.
    toggleSwitch.addEventListener('change', function() {
      chrome.storage.local.set({'isExtensionEnabled': toggleSwitch.checked});
    });
  });  
