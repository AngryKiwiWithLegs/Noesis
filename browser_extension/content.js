/**
 * noesis/browser_extension/content.js
 *
 * Monitors conversation DOM for new AI messages.
 * When detected, sends the turn to the local Noesis daemon via WebSocket.
 *
 * Tier 3 — capture only (cannot inject memories into web UIs).
 * Full bidirectional memory requires using an API-compatible tool (Tier 2).
 */

const NOESIS_WS_URL = "ws://localhost:8082/ingest";
const SITE_CONFIG = {
  "chat.openai.com": {
    name:        "chatgpt-web",
    userSel:     '[data-message-author-role="user"]',
    assistSel:   '[data-message-author-role="assistant"]',
    waitMs:      800,
  },
  "claude.ai": {
    name:        "claude-web",
    userSel:     ".font-user-message, [data-testid='user-message']",
    assistSel:   ".font-claude-message, [data-testid='assistant-message']",
    waitMs:      600,
  },
  "gemini.google.com": {
    name:        "gemini-web",
    userSel:     ".user-query",
    assistSel:   ".model-response-text, .markdown",
    waitMs:      1000,
  },
  "www.perplexity.ai": {
    name:        "perplexity-web",
    userSel:     ".prose.dark\\:prose-invert",
    assistSel:   ".prose:not(.dark\\:prose-invert)",
    waitMs:      1200,
  },
};

// ── Setup ─────────────────────────────────────────────────────────────────────

const site   = SITE_CONFIG[location.hostname];
if (!site) {
  // Not a supported site — do nothing
  throw new Error("Noesis: unsupported site");
}

let ws          = null;
let lastSeen    = "";   // hash of last sent turn (dedup)
let sessionId   = `ws-${Date.now()}`;

function connectWS() {
  ws = new WebSocket(NOESIS_WS_URL);
  ws.onopen    = () => console.log("Noesis: connected to daemon");
  ws.onerror   = () => { /* daemon not running — silent fail */ };
  ws.onclose   = () => {
    ws = null;
    // Retry after 30s if tab still open
    setTimeout(connectWS, 30_000);
  };
}

connectWS();

// ── DOM monitoring ────────────────────────────────────────────────────────────

let debounce = null;

const observer = new MutationObserver(() => {
  clearTimeout(debounce);
  // Wait for the AI response to finish streaming before capturing
  debounce = setTimeout(captureLatestTurn, site.waitMs);
});

observer.observe(document.body, { childList: true, subtree: true });

function captureLatestTurn() {
  try {
    const userMsgs   = [...document.querySelectorAll(site.userSel)];
    const assistMsgs = [...document.querySelectorAll(site.assistSel)];

    if (!userMsgs.length || !assistMsgs.length) return;

    const lastUser  = userMsgs[userMsgs.length - 1]?.innerText?.trim()   || "";
    const lastAssist = assistMsgs[assistMsgs.length - 1]?.innerText?.trim() || "";

    if (!lastUser || !lastAssist) return;

    const turnHash = btoa(encodeURIComponent(lastUser + lastAssist)).slice(0, 16);
    if (turnHash === lastSeen) return;   // already sent
    lastSeen = turnHash;

    sendTurn(lastUser, lastAssist);
  } catch (e) {
    console.debug("Noesis capture error:", e);
  }
}

// ── Send to daemon ────────────────────────────────────────────────────────────

function sendTurn(userMsg, aiMsg) {
  const payload = JSON.stringify({
    user_message: userMsg,
    ai_message:   aiMsg,
    source_tool:  site.name,
    session_id:   sessionId,
    user_id:      getUserId(),
    timestamp:    Date.now(),
  });

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(payload);
    showBadge();
  }
  // If ws not ready, drop silently — don't block user
}

// ── User ID persistence ───────────────────────────────────────────────────────

function getUserId() {
  // Check localStorage for previously set user ID
  try {
    return localStorage.getItem("noesis_user_id") || "default";
  } catch {
    return "default";
  }
}

// ── Visual feedback ───────────────────────────────────────────────────────────

function showBadge() {
  // Brief visual indicator that a memory was captured
  let badge = document.getElementById("noesis-badge");
  if (!badge) {
    badge = document.createElement("div");
    badge.id = "noesis-badge";
    Object.assign(badge.style, {
      position:   "fixed",
      bottom:     "12px",
      right:      "12px",
      background: "rgba(0,100,80,0.85)",
      color:      "#fff",
      padding:    "4px 10px",
      borderRadius: "6px",
      fontSize:   "11px",
      zIndex:     "99999",
      transition: "opacity 0.5s",
      pointerEvents: "none",
    });
    document.body.appendChild(badge);
  }
  badge.innerText = "⬡ memory captured";
  badge.style.opacity = "1";
  setTimeout(() => { badge.style.opacity = "0"; }, 2000);
}
