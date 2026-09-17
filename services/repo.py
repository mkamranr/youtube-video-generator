"""Fetch a GitHub repository or a Hugging Face model/dataset/space as script source text.

The README (or model card) plus the project's real facts - stars, downloads, licence, task,
library - become the text a video script is written from, the same way a YouTube transcript does
in services/transcript.py. Stdlib + requests only: `bs4` and `markdown` are importable on some
dev machines but are NOT in requirements.txt, so they do not exist in the container.
"""
import html
import os
import re
from urllib.parse import urlparse

import requests

import config

UA = "videogen/1.0 (+https://github.com/mkamranr/youtube-video-generator)"
TIMEOUT = 20                 # per request, connect + read
MIN_WORDS = 120              # below this there is not enough material to write a script from
MAX_README_BYTES = 400_000   # past this a README is vendored docs and changelogs, not a story

# path segments that start a sub-route of the repo rather than being part of its id
_TAIL = {"tree", "blob", "raw", "resolve", "commit", "commits", "discussions", "issues",
         "pull", "pulls", "releases", "tags", "wiki", "actions", "settings", "files",
         "blame", "compare", "graphs", "network", "stargazers", "forks", "branches"}
_HF_TYPES = {"models": "hf_model", "datasets": "hf_dataset", "spaces": "hf_space"}
_HF_PREFIX = {"hf_model": "", "hf_dataset": "datasets/", "hf_space": "spaces/"}
_HF_API = {"hf_model": "models", "hf_dataset": "datasets", "hf_space": "spaces"}
_KIND_LABEL = {"github": "GitHub repository", "hf_model": "Hugging Face model",
               "hf_dataset": "Hugging Face dataset", "hf_space": "Hugging Face space"}


# ---- url parsing -----------------------------------------------------------
def parse_url(url: str) -> dict:
    """{kind, id, owner, name, page_url} for a GitHub repo or a HF model/dataset/space URL.

    Sub-routes (/tree/main/src, /blob/..., /issues) are trimmed back to the repository itself.
    Raises ValueError for anything else, so a bad URL is rejected at submit time instead of
    failing later inside the worker (which is what the YouTube path does today)."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("enter a GitHub or Hugging Face URL")
    m = re.match(r"^git@([^:]+):(.+)$", raw)            # git@github.com:owner/repo.git
    if m:
        raw = f"https://{m.group(1)}/{m.group(2)}"
    if "//" not in raw:
        raw = "https://" + raw
    u = urlparse(raw)
    host = (u.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    parts = [p for p in u.path.split("/") if p]
    cut = next((i for i, p in enumerate(parts) if p.lower() in _TAIL), len(parts))
    parts = parts[:cut]

    if host == "github.com":
        if len(parts) < 2:
            raise ValueError("a GitHub URL must point at a repository, "
                             "e.g. https://github.com/owner/repo")
        owner, name = parts[0], re.sub(r"\.git$", "", parts[1])
        if not owner or not name:
            raise ValueError("a GitHub URL must point at a repository, "
                             "e.g. https://github.com/owner/repo")
        return {"kind": "github", "id": f"{owner}/{name}", "owner": owner, "name": name,
                "page_url": f"https://github.com/{owner}/{name}"}

    if host in ("huggingface.co", "hf.co"):
        kind = "hf_model"
        if parts and parts[0] in _HF_TYPES:
            kind = _HF_TYPES[parts[0]]
            parts = parts[1:]
        if not parts:
            raise ValueError("a Hugging Face URL must point at a model, dataset or space, "
                             "e.g. https://huggingface.co/owner/model")
        rid = "/".join(parts[:2])
        return {"kind": kind, "id": rid, "owner": parts[0] if len(parts) > 1 else "",
                "name": parts[-1],
                "page_url": f"https://huggingface.co/{_HF_PREFIX[kind]}{rid}"}

    raise ValueError("not a GitHub or Hugging Face URL - paste a github.com repository or a "
                     "huggingface.co model, dataset or space link")


def normalize_url(url: str) -> str:
    """The canonical page URL; raises ValueError for anything that is not one of the four kinds."""
    return parse_url(url)["page_url"]


# ---- http ------------------------------------------------------------------
def _github_headers(accept: str) -> dict:
    h = {"User-Agent": UA, "Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url: str, accept: str = "application/json") -> requests.Response:
    """GET with a User-Agent (GitHub requires one) and human-readable errors.

    The status codes these two APIs really use are not the obvious ones: a GitHub repo with no
    README 404s exactly like a missing repo, a gated HF model answers 200 from the API but 401
    from the README (with a plain-text body, so .json() would throw), and an unknown HF space
    answers 401 rather than 404."""
    github = "api.github.com" in url
    headers = _github_headers(accept) if github else {"User-Agent": UA, "Accept": accept}
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    if github and r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
        reset = r.headers.get("X-RateLimit-Reset", "")
        when = ""
        try:
            import time
            mins = max(0, int((int(reset) - time.time()) / 60))
            when = f" (resets in about {mins} minute{'s' if mins != 1 else ''})"
        except (TypeError, ValueError):
            pass
        raise RuntimeError("GitHub API rate limit reached - 60 requests per hour are shared by "
                           f"everyone on this network{when}. Set GITHUB_TOKEN to raise it to 5000.")
    if r.status_code == 404:
        raise RuntimeError("not found (404)")
    if r.status_code in (401, 403):
        raise RuntimeError(f"access denied ({r.status_code}) - the project may be private, gated "
                           "or require accepting its licence")
    r.raise_for_status()
    return r


def _readme(candidates: list) -> tuple:
    """First candidate URL that yields usable text -> (text, ""). None of them -> ("", why).

    Layered with accumulated errors, like services.transcript.get_transcript_info."""
    errors = []
    for url, accept in candidates:
        try:
            r = _get(url, accept)
            body = r.content[:MAX_README_BYTES].decode("utf-8", "replace")
            if len(body.strip()) > 40:
                return body, ""
            errors.append(f"{url}: empty")
        except Exception as e:  # noqa: BLE001 - try the next candidate, report them all
            errors.append(f"{url}: {e}")
    return "", " | ".join(errors)


# ---- markdown -> prose -----------------------------------------------------
def markdown_to_text(md: str) -> str:
    """Readable prose from a README or model card. Regex only - order matters here."""
    t = (md or "").replace("\r\n", "\n")
    m = re.match(r"^---\n.*?\n---\n", t, re.S)                       # HF card YAML front matter
    if m:
        t = t[m.end():]
    t = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "\n", t, flags=re.S | re.M)    # fenced code
    t = re.sub(r"^~~~[^\n]*\n.*?^~~~[ \t]*$", "\n", t, flags=re.S | re.M)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]{0,400}>", " ", t)                 # bounded: an unclosed '<' cannot hang
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)          # images and badges: drop entirely
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)       # links: keep the text
    t = re.sub(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", "", t, flags=re.M)   # reference definitions
    t = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", t)         # inline code
    t = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]*", "", t, flags=re.M)          # heading markers
    t = re.sub(r"^[ \t]{0,3}([-*_][ \t]*){3,}$", "", t, flags=re.M)    # horizontal rules
    t = re.sub(r"^[ \t]*\|.*\|[ \t]*$", "", t, flags=re.M)             # table rows
    t = re.sub(r"^[ \t]{0,3}>[ \t]?", "", t, flags=re.M)               # block quotes
    t = re.sub(r"^[ \t]{0,6}[-*+][ \t]+", "- ", t, flags=re.M)         # bullets, normalised
    t = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", t)                   # bold / italic
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---- fact blocks -----------------------------------------------------------
def _facts_github(m: dict) -> list:
    lic = ((m.get("license") or {}).get("spdx_id") or "").replace("NOASSERTION", "")
    return [f"GitHub repository: {m.get('full_name') or ''}",
            f"Description: {m['description']}" if m.get("description") else "",
            f"Primary language: {m['language']}" if m.get("language") else "",
            f"Topics: {', '.join(m.get('topics') or [])}" if m.get("topics") else "",
            f"Stars: {m.get('stargazers_count', 0):,} - Forks: {m.get('forks_count', 0):,} - "
            f"Open issues: {m.get('open_issues_count', 0):,}",
            f"License: {lic}" if lic else "",
            f"Homepage: {m['homepage']}" if m.get("homepage") else "",
            f"Created {(m.get('created_at') or '')[:10]}, last pushed "
            f"{(m.get('pushed_at') or '')[:10]}" if m.get("pushed_at") else "",
            "This repository is archived and no longer maintained." if m.get("archived") else ""]


def _tagged(tags: list, prefix: str) -> list:
    return [t.split(":", 1)[1] for t in (tags or []) if t.startswith(prefix + ":")]


def _facts_hf(kind: str, m: dict) -> list:
    card, tags = (m.get("cardData") or {}), (m.get("tags") or [])
    out = [f"{_KIND_LABEL[kind]}: {m.get('id') or ''}",
           f"Title: {card['title']}" if card.get("title") else "",
           f"Summary: {card['short_description']}" if card.get("short_description") else "",
           f"Task: {m['pipeline_tag']}" if m.get("pipeline_tag") else "",
           f"Library: {m['library_name']}" if m.get("library_name") else "",
           f"SDK: {card['sdk']}" if card.get("sdk") else "",
           f"Downloads (30 days): {m.get('downloads', 0):,} - Likes: {m.get('likes', 0):,}",
           f"Last updated: {(m.get('lastModified') or '')[:10]}" if m.get("lastModified") else "",
           "This repository is gated: access must be requested." if m.get("gated") else ""]
    for label, pre in (("Tasks", "task_categories"), ("Size", "size_categories"),
                       ("Languages", "language"), ("License", "license"),
                       ("Trained on", "dataset"), ("Base model", "base_model")):
        vals = _tagged(tags, pre)
        if vals:
            out.append(f"{label}: {', '.join(vals[:8])}")
    free = [t for t in tags if ":" not in t][:12]
    out.append(f"Tags: {', '.join(free)}" if free else "")
    return out


def build_text(facts: list, readme: str) -> str:
    """The fact block, then the cleaned README, capped at the same 60k chars a transcript is."""
    body = markdown_to_text(readme)
    blob = "\n".join(f for f in facts if f)
    if body:
        blob += "\n\nREADME / MODEL CARD\n\n" + body
    return blob[: config.MAX_TRANSCRIPT_CHARS]


# ---- entry point -----------------------------------------------------------
def fetch(url: str) -> dict:
    """{kind, id, owner, name, page_url, text, meta, words} for a repo / model / dataset / space."""
    info = parse_url(url)
    kind, rid = info["kind"], info["id"]

    if kind == "github":
        try:
            meta = _get(f"https://api.github.com/repos/{rid}").json()
        except RuntimeError as e:
            if "not found" in str(e):
                raise RuntimeError(f"This repository cannot be read: {rid} does not exist or is "
                                   "private on GitHub.")
            raise
        info["id"] = meta.get("full_name") or rid          # follows renames / redirects
        info["page_url"] = meta.get("html_url") or info["page_url"]
        readme, why = _readme([(f"https://api.github.com/repos/{info['id']}/readme",
                                "application/vnd.github.raw+json")])
    else:
        try:
            meta = _get(f"https://huggingface.co/api/{_HF_API[kind]}/{rid}").json()
        except RuntimeError as e:
            raise RuntimeError(f"This {_KIND_LABEL[kind].lower()} cannot be read: {rid} "
                               f"({e}). It may not exist, or it may be private or gated.")
        canon = meta.get("id") or rid                      # short ids resolve to owner/name
        pre = _HF_PREFIX[kind]
        info["id"] = canon
        info["page_url"] = f"https://huggingface.co/{pre}{canon}"
        readme, why = _readme([(f"https://huggingface.co/{pre}{canon}/raw/main/README.md", "text/plain"),
                               (f"https://huggingface.co/{pre}{canon}/raw/main/readme.md", "text/plain")])

    facts = _facts_github(meta) if kind == "github" else _facts_hf(kind, meta)
    text = build_text(facts, readme)
    words = len(text.split())
    if words < MIN_WORDS:
        raise RuntimeError(
            f"This {_KIND_LABEL[kind].lower()} cannot be turned into a video: only {words} words of "
            f"README and facts were found (at least {MIN_WORDS} are needed to write a script). "
            "Pick a project with a real README, or use Description mode."
            + (f" README: {why}" if why else ""))
    meta["page_url"] = info["page_url"]
    return {**info, "text": text, "meta": meta, "words": words}
