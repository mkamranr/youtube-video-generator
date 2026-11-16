"""Reel format: the structured, result-first script of a vertical short.

Pure Python, no LLM and no ffmpeg, so every rule here is cheap to test. The LLM writes the
candidates (services/llm.write_reel_script); this module decides which hook wins, whether the
script is fit to speak, how the scenes become the one narration paragraph the rest of the
pipeline reads, and which scene each rendered clip belongs to.

plan["reel"] shape:
    {"hooks": [{"text", "category", "scores": {SCORE_KEYS: 1-5}, "supported": bool,
                "rejected": "" | reason}],
     "chosen": int,                       # index into hooks, -1 when none survived
     "scenes": [{"n", "role", "voiceover", "headline", "support_text", "visual",
                 "asset_needed", "transition", "emphasis": [], "has_claim", "asset": ""}],
     "cta": {"type", "text"},
     "central_message": str, "audience_benefit": str,
     "script_version": int, "issues": [str]}
"""
from __future__ import annotations

import re

SCORE_KEYS = ("clarity", "relevance", "curiosity", "specificity", "factual", "accessibility")
HOOK_CATEGORIES = ("outcome", "surprise", "problem_solution", "curiosity", "comparison",
                   "demo_first", "list")
ROLES = ("hook", "context", "demo", "payoff", "cta")
CTA_TYPES = ("save", "share", "question", "follow", "repo")
MIN_HOOKS = 5
MIN_FACTUAL = 4              # factual support is mandatory, whatever the other scores say
LENGTH_TOLERANCE = 0.15      # the script may land this far either side of its word target

# Hype the spec bans outright, and claims that need the source to back them. The second group is
# only a reason to reject when the hook is not marked as supported - the LLM is told to mark it.
HYPE_WORDS = ("revolutionary", "unlimited", "best ever", "game changer", "game-changer",
              "mind-blowing", "insane", "completely free", "100% free", "完全")
CLAIM_PATTERNS = (r"\bfree\b", r"\boffline\b", r"\bon (?:a|your|any) laptop\b",
                  r"\bopen[- ]source\b", r"\bprivate\b")
BANNED_OPENERS = (r"^\s*today,? we(?:'re| are| will)", r"^\s*in this video", r"^\s*hey\b",
                  r"^\s*hi\b", r"^\s*hello\b", r"^\s*welcome\b", r"^\s*what'?s up\b",
                  r"^\s*let'?s (?:take a look|look at|talk about|dive)")

# the asks a call to action can make; more than one kind in the CTA scene is asking too much
CTA_ASKS = {"save": r"\bsave\b", "share": r"\bshare\b|\bsend (?:this|it)\b",
            "follow": r"\bfollow\b|\bsubscribe\b", "comment": r"\bcomment\b|\blet me know\b",
            "repo": r"\bcheck (?:out )?the (?:repo|repository|project|link)\b|\blink in\b",
            "like": r"\blike (?:this|the video)\b"}

CTA_TEXTS = {
    "save": "Save this so you can try it later.",
    "share": "Share this with a developer who would find it useful.",
    "question": "",
    "follow": "Follow Daily Tech Talks for more practical AI tools.",
    "repo": "Check out the project - link in the caption.",
}


def _int(v, lo=1, hi=5, default=1) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def _str(v, limit=600) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:limit]


def total(hook: dict) -> int:
    return sum(int(hook.get("scores", {}).get(k, 0)) for k in SCORE_KEYS)


def hype_reason(text: str, supported: bool) -> str:
    """Why a line may not be used as a hook, or "" when it is fine."""
    low = text.lower()
    for w in HYPE_WORDS:
        if w in low:
            return f'uses hype word "{w}"'
    if not supported:
        for pat in CLAIM_PATTERNS:
            m = re.search(pat, low)
            if m:
                return f'claims "{m.group(0)}" without support from the source'
    return ""


def normalize_hooks(raw) -> list:
    out = []
    for h in raw if isinstance(raw, list) else []:
        if isinstance(h, str):
            h = {"text": h}
        if not isinstance(h, dict):
            continue
        text = _str(h.get("text"), 200)
        if not text:
            continue
        cat = _str(h.get("category"), 40).lower().replace("/", "_").replace("-", "_").replace(" ", "_")
        scores = h.get("scores") if isinstance(h.get("scores"), dict) else {}
        out.append({"text": text,
                    "category": cat if cat in HOOK_CATEGORIES else "outcome",
                    "scores": {k: _int(scores.get(k)) for k in SCORE_KEYS},
                    "supported": bool(h.get("supported", True)),
                    "rejected": ""})
    return out


def select_hook(hooks: list) -> int:
    """Mark the hooks that may not be used and return the index of the best remaining one
    (-1 when none survive). Highest total wins; clarity breaks a tie, then the earlier hook."""
    best, key = -1, None
    for i, h in enumerate(hooks):
        reason = ""
        if not h.get("supported", True):
            reason = "not supported by the source"
        elif int(h["scores"].get("factual", 0)) < MIN_FACTUAL:
            reason = f"factual support below {MIN_FACTUAL}"
        else:
            reason = hype_reason(h["text"], h.get("supported", True))
        h["rejected"] = reason
        if reason:
            continue
        k = (total(h), int(h["scores"].get("clarity", 0)), -i)
        if key is None or k > key:
            best, key = i, k
    return best


def normalize_scenes(raw) -> list:
    out = []
    for i, s in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(s, dict):
            continue
        vo = _str(s.get("voiceover"), 1200)
        if not vo:
            continue
        role = _str(s.get("role"), 20).lower()
        emph = s.get("emphasis") if isinstance(s.get("emphasis"), list) else []
        out.append({"n": len(out) + 1,
                    "role": role if role in ROLES else ("hook" if i == 0 else "context"),
                    "voiceover": vo,
                    "headline": _str(s.get("headline"), 60),
                    "support_text": _str(s.get("support_text"), 120),
                    "visual": _str(s.get("visual"), 600),
                    "asset_needed": _str(s.get("asset_needed"), 200),
                    "transition": _str(s.get("transition"), 60),
                    "emphasis": [_str(e, 40) for e in emph if _str(e, 40)][:6],
                    "has_claim": bool(s.get("has_claim", False)),
                    "asset": _str(s.get("asset"), 80)})
    return out


def normalize_cta(raw, scenes: list) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    kind = _str(raw.get("type"), 20).lower()
    kind = kind if kind in CTA_TYPES else "save"
    text = _str(raw.get("text"), 140)
    if not text:
        last = scenes[-1] if scenes and scenes[-1]["role"] == "cta" else None
        text = last["voiceover"] if last else CTA_TEXTS.get(kind) or CTA_TEXTS["save"]
    return {"type": kind, "text": text}


def build(raw: dict, script_version: int = 1) -> dict:
    """Normalize the writer's JSON into plan["reel"] and pick the hook."""
    scenes = normalize_scenes(raw.get("scenes"))
    hooks = normalize_hooks(raw.get("hooks"))
    reel = {"hooks": hooks, "chosen": select_hook(hooks), "scenes": scenes,
            "cta": normalize_cta(raw.get("cta"), scenes),
            "central_message": _str(raw.get("central_message"), 300),
            "audience_benefit": _str(raw.get("audience_benefit"), 300),
            "script_version": int(script_version), "issues": []}
    apply_hook(reel)
    return reel


def apply_hook(reel: dict, index: int | None = None) -> dict:
    """Make hook `index` (default: the chosen one) the first spoken line of scene 1."""
    if index is not None:
        if not 0 <= index < len(reel["hooks"]):
            raise ValueError("no such hook")
        reel["chosen"] = index
    i = reel.get("chosen", -1)
    if i is None or i < 0 or i >= len(reel["hooks"]) or not reel["scenes"]:
        return reel
    text = reel["hooks"][i]["text"]
    first = reel["scenes"][0]
    prev = first.get("hook_text", "")
    vo = first["voiceover"]
    # the writer opens scene 1 with ITS favourite hook, which need not be the one that scored best:
    # whichever candidate the line starts with goes, so the video never speaks two hooks in a row
    for cand in [prev] + [h["text"] for h in reel["hooks"]]:
        if cand and vo.lower().startswith(cand.lower()):
            vo = vo[len(cand):].lstrip()
            break
    first["voiceover"] = (text + (" " + vo if vo else "")).strip()
    first["hook_text"] = text
    first["role"] = "hook"
    return reel


def narration(reel: dict) -> str:
    return " ".join(s["voiceover"] for s in reel.get("scenes") or [] if s.get("voiceover"))


def hook_headline(reel: dict | None) -> str:
    if not reel or not reel.get("scenes"):
        return ""
    first = reel["scenes"][0]
    if first.get("headline"):
        return first["headline"]
    i = reel.get("chosen", -1)
    return reel["hooks"][i]["text"] if 0 <= i < len(reel.get("hooks") or []) else ""


def _sentences(text: str) -> list:
    return [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


def validate(reel: dict, target_words: int | None) -> list:
    """Everything that should be fixed before the script is spoken; [] when it is fit."""
    issues = []
    scenes = reel.get("scenes") or []
    if not scenes:
        return ["the script has no scenes"]
    if len(reel.get("hooks") or []) < MIN_HOOKS:
        issues.append(f"only {len(reel.get('hooks') or [])} hook candidates (need {MIN_HOOKS})")
    if reel.get("chosen", -1) < 0:
        issues.append("no hook candidate is supported by the source without hype")
    if scenes[0]["role"] != "hook":
        issues.append("scene 1 must be the hook")
    first = scenes[0]["voiceover"].lower()
    for pat in BANNED_OPENERS:
        if re.search(pat, first):
            issues.append("the script opens with an introduction instead of the hook")
            break
    roles = [s["role"] for s in scenes]
    if "demo" not in roles:
        issues.append("no demonstration scene proves the main claim")
    ctas = roles.count("cta")
    if ctas != 1:
        issues.append(f"needs exactly one call to action, found {ctas}")
    elif roles[-1] != "cta":
        issues.append("the call to action must be the last scene")
    else:
        asks = [k for k, pat in CTA_ASKS.items() if re.search(pat, scenes[-1]["voiceover"].lower())]
        if len(asks) > 1:
            issues.append(f"the call to action asks for several things ({', '.join(asks)}) - keep one")
    if not reel.get("central_message"):
        issues.append("no central message")
    words = len(narration(reel).split())
    if target_words:
        lo, hi = target_words * (1 - LENGTH_TOLERANCE), target_words * (1 + LENGTH_TOLERANCE)
        if words > hi:
            issues.append(f"too long: {words} words for a {target_words}-word target")
        elif words < lo:
            issues.append(f"too short: {words} words for a {target_words}-word target")
    seen = set()
    for s in scenes:
        for sent in _sentences(s["voiceover"]):
            if sent in seen:
                issues.append(f'repeats the line "{sent[:60]}"')
                break
            seen.add(sent)
    return issues


def apply_edits(reel: dict, edits: dict) -> dict:
    """Merge the approval page's edits (chosen hook, scene text, CTA text) into the reel.

    The hook swap is applied first and any edited voiceover after it: the page already rewrote
    scene 1 with the new hook, so a voiceover it sends is final and must not get the hook again."""
    if not isinstance(edits, dict):
        return reel
    if "chosen" in edits and edits["chosen"] is not None:
        idx = _int(edits["chosen"], -1, 999, -1)
        if idx != reel.get("chosen") and 0 <= idx < len(reel.get("hooks") or []):
            apply_hook(reel, idx)
    by_n = {s["n"]: s for s in reel.get("scenes") or []}
    for e in edits.get("scenes") or []:
        if not isinstance(e, dict):
            continue
        s = by_n.get(_int(e.get("n"), 1, 999, 0))
        if not s:
            continue
        for k, limit in (("voiceover", 1200), ("headline", 60), ("support_text", 120)):
            if k in e:
                s[k] = _str(e[k], limit)
    reel["scenes"] = [s for s in reel.get("scenes") or [] if s.get("voiceover")]
    for i, s in enumerate(reel["scenes"]):
        s["n"] = i + 1
    if isinstance(edits.get("cta"), dict) and _str(edits["cta"].get("text"), 140):
        reel["cta"]["text"] = _str(edits["cta"]["text"], 140)
    return reel


def clip_scene_map(scenes: list, clip_seconds: list, audio_seconds: float | None = None) -> list:
    """For each clip, (scene index, the words spoken during it), estimating word times from their
    position in the narration - the narration is spoken at an even pace, and this runs before the
    whisper word timings exist. Each clip gets the scene whose words fill most of it."""
    words = []                      # (scene index, word)
    for si, s in enumerate(scenes):
        words += [(si, w) for w in s["voiceover"].split()]
    n = len(clip_seconds)
    if not words or not n:
        return [(0, "")] * n
    spoken = float(audio_seconds or 0) or float(sum(clip_seconds)) or 1.0
    per = len(words) / spoken
    out, t0 = [], 0.0
    for secs in clip_seconds:
        a, b = int(round(t0 * per)), int(round((t0 + secs) * per))
        chunk = words[a:max(b, a)]
        if chunk:
            counts = {}
            for si, _ in chunk:
                counts[si] = counts.get(si, 0) + 1
            scene = max(counts, key=lambda k: (counts[k], -k))
        else:   # past the last word (music tail): stay on the last scene
            scene = words[min(a, len(words) - 1)][0]
        out.append((scene, " ".join(w for _, w in chunk)))
        t0 += secs
    return out


OPENING_RULE = ("This is the reel's OPENING clip: the subject must be clear and recognizable in the very "
                "first frame - the result, the interface or the hardware itself, filling the frame. No "
                "fade-in, no empty or dark frame, no logo reveal, no slow establishing shot.")
ROLE_RULES = {
    "hook": "Show the most striking result or object straight away, big in frame.",
    "context": "Show why it matters: the everyday problem or the situation it changes.",
    "demo": ("Show the actual thing working: a screen with the interface or output clearly visible, "
             "close enough to read on a phone, the camera pushing in on the key detail."),
    "payoff": "Show the result in use or the final outcome, calm and clear.",
    "cta": "A clean, uncluttered closing shot that leaves room for the end card.",
}


def clip_hints(scenes: list, clip_scene: list) -> list:
    """One direction per clip from the reel scene it belongs to: what the scene's role needs on
    screen and the writer's own visual instruction. Clip 1 also carries the opening rule."""
    out = []
    for ci, si in enumerate(clip_scene):
        sc = scenes[si] if 0 <= si < len(scenes) else {}
        parts = []
        if ci == 0:
            parts.append(OPENING_RULE)
        parts.append(f"This clip belongs to scene {sc.get('n', si + 1)} ({sc.get('role', '')}). "
                     + ROLE_RULES.get(sc.get("role", ""), ""))
        if sc.get("visual"):
            parts.append(f"On screen: {sc['visual']}")
        out.append(" ".join(p.strip() for p in parts if p.strip()))
    return out


def social_text(social) -> str:
    """The reel's platform captions as a plain-text block for the bundle ("" when none)."""
    if not isinstance(social, dict) or not social:
        return ""
    ig, fb, yt = (social.get(k) or {} for k in ("instagram", "facebook", "youtube"))
    out = "\r\n==== Instagram ====\r\n" + (ig.get("caption") or "")
    if ig.get("hashtags"):
        out += "\r\n\r\n" + " ".join(ig["hashtags"])
    out += "\r\n\r\n==== Facebook ====\r\n" + (fb.get("caption") or "")
    out += ("\r\n\r\n==== YouTube Shorts ====\r\nTitle: " + (yt.get("title") or "")
            + "\r\n" + (yt.get("description") or "") + "\r\n")
    return out
