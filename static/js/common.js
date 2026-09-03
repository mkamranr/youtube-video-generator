/* VideoGen — shared helpers: theme, icons, navbar, status utilities */
const $ = s => document.querySelector(s);

/* ---- theme (dark default, persisted) ---- */
(function initTheme() {
  let t = "dark";
  try { t = localStorage.getItem("videogen-theme") || "dark"; } catch (e) { /* ignore */ }
  document.documentElement.dataset.theme = t;
})();

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("videogen-theme", next); } catch (e) { /* ignore */ }
  const btn = $("#themeToggle");
  if (btn) btn.innerHTML = icon(next === "light" ? "moon" : "sun");
}

/* ---- inline SVG icons ---- */
const ICONS = {
  play: '<path d="M8 5.5v13l11-6.5z" fill="currentColor" stroke="none"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1.2" fill="currentColor" stroke="none"/><circle cx="4.5" cy="12" r="1.2" fill="currentColor" stroke="none"/><circle cx="4.5" cy="18" r="1.2" fill="currentColor" stroke="none"/>',
  archive: '<rect x="3" y="4" width="18" height="5" rx="1"/><path d="M5 9v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9M10 13h4"/>',
  gear: '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v2.6M12 18.6v2.6M2.8 12h2.6M18.6 12h2.6M5.5 5.5l1.9 1.9M16.6 16.6l1.9 1.9M18.5 5.5l-1.9 1.9M7.4 16.6l-1.9 1.9"/>',
  sun: '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.3M12 19.2v2.3M2.5 12h2.3M19.2 12h2.3M4.9 4.9l1.7 1.7M17.4 17.4l1.7 1.7M19.1 4.9l-1.7 1.7M6.6 17.4l-1.7 1.7"/>',
  moon: '<path d="M20 13.5A8 8 0 1 1 10.5 4 6.5 6.5 0 0 0 20 13.5z"/>',
  check: '<path d="M4.5 12.5l5 5L19.5 7"/>',
  refresh: '<path d="M20 11.5A8 8 0 1 0 18.5 16"/><path d="M20 6v5.5h-5.5"/>',
  video: '<rect x="2.5" y="6" width="13" height="12" rx="2.5"/><path d="M15.5 12l6-3.8v7.6z"/>',
  back: '<path d="M14.5 5.5L8 12l6.5 6.5"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
  upload: '<path d="M12 15.5V4.5M7 9l5-4.5L17 9"/><path d="M4.5 19.5h15"/>',
};

function icon(name) {
  return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;
}

/* ---- navbar ---- */
function navbar(active) {
  const theme = document.documentElement.dataset.theme;
  return `<nav class="nav">
    <a class="brand" href="/"><span class="logo" aria-hidden="true"></span> VideoGen</a>
    <div class="links">
      <a href="/" class="${active === "home" ? "active" : ""}">${icon("list")} Jobs</a>
      <a href="/archive" class="${active === "archive" ? "active" : ""}">${icon("archive")} Archived Jobs</a>
      <a href="/settings" class="${active === "settings" ? "active" : ""}">${icon("gear")} Settings</a>
    </div>
    <div class="spacer"></div>
    <button class="icon-btn" id="themeToggle" title="Toggle dark/light theme" onclick="toggleTheme()">${icon(theme === "light" ? "moon" : "sun")}</button>
  </nav>`;
}

/* ---- status helpers ---- */
const STEP_LABELS = { queued: "Queued", drafting: "Script", awaiting_approval: "Approval", transcript: "Transcript", planning: "LLM plan", tts: "Narration", scenes: "Scenes", thumbnail: "Thumbnail", music: "Music", clips: "Clips", assembling: "Assemble", done: "Done" };

function orderFor(j) {
  const mid = j.audio_mode === "music" ? ["music", "scenes", "thumbnail"]
            : j.audio_mode === "narration" ? ["tts", "scenes", "thumbnail"]
            : ["tts", "scenes", "thumbnail", "music"];
  const head = (j.source === "topic" || j.source === "description")
            ? ["drafting", "awaiting_approval", "queued", "planning"]
            : ["queued", "transcript", "planning"];
  return [...head, ...mid, "clips", "assembling", "done"];
}

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function pct(j) {
  if (j.status === "done") return 100;
  if (j.status === "failed") return 0;
  if (j.status === "clips") return j.clips_total ? Math.round(40 + 52 * j.clips_done / j.clips_total) : 40;
  if (j.status === "assembling") return 95;
  const order = orderFor(j);
  const i = order.indexOf(j.status), clipsIdx = order.indexOf("clips");
  return i < 0 ? 0 : Math.round(2 + 38 * i / clipsIdx);
}

function pillHTML(j) {
  const active = !["done", "failed", "queued", "awaiting_approval"].includes(j.status);
  const cls = j.status === "done" ? "done" : j.status === "failed" ? "failed"
            : j.status === "queued" ? "queued" : j.status === "awaiting_approval" ? "approval" : "running";
  const text = j.status === "clips" && j.clips_total ? `clips ${j.clips_done}/${j.clips_total}`
             : j.status === "awaiting_approval" ? "review script" : (STEP_LABELS[j.status] || j.status);
  return `<span class="pill ${cls} ${active ? "running" : ""}">${esc(text.toLowerCase())}</span>`;
}

function timeAgo(ts) {
  const s = Math.max(0, (Date.now() / 1000) - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

let toastTimer = null;
function toast(msg, isError) {
  let el = $("#toast");
  if (!el) { el = document.createElement("div"); el.id = "toast"; document.body.appendChild(el); }
  el.textContent = msg;
  el.classList.toggle("error", !!isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}
