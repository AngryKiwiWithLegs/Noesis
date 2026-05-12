/**
 * browser_extension/background.js
 * Manifest V3 service worker — minimal, just handles install lifecycle.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.log("Noesis extension installed.");
  // Set default user ID
  chrome.storage.local.get(["noesisUserId"], (r) => {
    if (!r.noesisUserId) {
      chrome.storage.local.set({ noesisUserId: "default" });
    }
  });
});

// Keep service worker alive during active connections
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "ping") sendResponse({ pong: true });
});
