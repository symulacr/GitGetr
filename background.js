// Handle runtime events and permissions management
chrome.runtime.onInstalled.addListener(() => {
  // Add necessary permissions
  chrome.permissions.request({
    permissions: [
      'storage',
      'activeTab',
      'downloads',
      'https://api.github.com/*' 
      // Add other necessary permissions based on specific functionalities
    ],
    origins: ['https://github.com/*']
    // Add more origins if content scripts need to be injected into other domains
  });
});

// Register content scripts for different domains
chrome.runtime.onInstalled.addListener(() => {
  chrome.declarativeContent.onPageChanged.removeRules(undefined, () => {
    chrome.declarativeContent.onPageChanged.addRules([
      {
        conditions: [
          new chrome.declarativeContent.PageStateMatcher({
            pageUrl: { hostEquals: 'github.com' }
            // Add more conditions for other domains if needed
          })
        ],
        actions: [new chrome.declarativeContent.ShowPageAction()]
      }
    ]);
  });
});

// Adjust content security policy based on extension requirements
chrome.runtime.onStartup.addListener(() => {
  const newCsp = {
    extension_pages: "script-src 'self'; object-src 'self'",
    sandbox: ["sandbox allow-scripts allow-forms"]
    // Adjust the content security policy based on extension requirements for specific functionalities
  };
  chrome.runtime.setComponentSettings({ csp: newCsp });
});

// Additional event listeners or tasks can be added based on extension requirements
// For example, handling messages, managing permissions dynamically, etc.
