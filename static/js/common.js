/* VideoGen — shared helpers */
const $ = s => document.querySelector(s);
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

function navbar(active) {
  return `<nav class="nav">
    <a class="brand" href="/"><span class="logo">▶</span> VideoGen</a>
    <div class="links">
      <a href="/" class="${active === "home" ? "active" : ""}">Jobs</a>
      <a href="/archive" class="${active === "archive" ? "active" : ""}">Archived Jobs</a>
      <a href="/settings" class="${active === "settings" ? "active" : ""}">Settings</a>
    </div>
    <div class="spacer"></div>
  </nav>`;
}
