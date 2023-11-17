// Example content.js

// Functionality for GitHub domain
if (window.location.hostname === 'github.com') {
  // Your code specific to GitHub functionality goes here
  // For instance, extracting media file links or interacting with GitHub's DOM elements
}

// Functionality for other domains (if applicable)
// Add additional if statements for different domains
// For example:
/*
else if (window.location.hostname === 'example.com') {
  // Your code specific to 'example.com' functionality goes here
}
*/

// Permissions (if needed for certain actions)
// Example: requesting permission to access the current tab
chrome.permissions.request({
  permissions: ['tabs'],
  origins: [window.location.origin]
}, function(granted) {
  if (granted) {
    // Permission granted, you can perform actions here
  } else {
    // Permission not granted, handle accordingly
  }
});

// Adjusting content security policy (if necessary)
// Example: modifying content security policy
const meta = document.createElement('meta');
meta.httpEquiv = 'Content-Security-Policy';
meta.content = "script-src 'self' https://example.com; object-src 'self'";
document.head.appendChild(meta);
