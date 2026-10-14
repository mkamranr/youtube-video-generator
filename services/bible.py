"""Character Bible import: turn a markdown "character bible" into channel data.

A bible is the document a channel author writes once and used to paste in front of every
story request: a global visual style, permanent character rules, one `### Name` block per
character with `- **Field:** value` bullets, episode rules, a consistency checklist. Here it
is parsed into

  * cast entries  {name, kind, role, look, personality, priority}  for services.channels (one per
    `###` block found in a section that has such blocks), and
  * a plain-text `rules` block for the channel's fixed "style & rules" text (every other `##`
    section, except a "scene/prompt format" section - the app has its own scene prompt format
    tuned for the video model).

The look is rebuilt as labelled sentences ("Precise look: ... Clothes: ... Distinguishing
marks: ...") so the labels survive into the deterministic ComfyUI look prefix. Pure text
processing: no project imports, so it is trivially testable.
"""
import re

_H2 = re.compile(r"^\s{0,3}##\s+(.*?)\s*#*\s*$")
_H3 = re.compile(r"^\s{0,3}###\s+(.*?)\s*#*\s*$")
_FIELD = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*(?:\*\*|__)?\s*([A-Za-z][A-Za-z /&'-]{0,40}?)\s*(?:\*\*|__)?"
                    r"\s*:\s*(?:\*\*|__)?\s*(.*)$")
_LIST_MARK = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")
_NUMBERED_TITLE = re.compile(r"^\s*\d+[.)]\s*")
_FORMAT_SECTION = re.compile(r"prompt\s*format|scene\s*format", re.IGNORECASE)
_PAREN_KIND = re.compile(r"^(.*?)\s*\(([^()]{1,80})\)\s*$")

# canonical look labels, in the order they are written into the look text
LOOK_ORDER = ("Precise look", "Details", "Clothes", "Accessories", "Distinguishing marks")
_LOOK_ALIASES = {
    "precise look": "Precise look", "look": "Precise look", "looks": "Precise look",
    "appearance": "Precise look", "description": "Precise look", "visual": "Precise look",
    "details": "Details", "detail": "Details",
    "clothes": "Clothes", "clothing": "Clothes", "outfit": "Clothes", "wears": "Clothes", "wearing": "Clothes",
    "accessories": "Accessories", "accessory": "Accessories",
    "distinguishing marks": "Distinguishing marks", "distinguishing mark": "Distinguishing marks",
    "distinguishing features": "Distinguishing marks", "marks": "Distinguishing marks",
    "markings": "Distinguishing marks", "unique marks": "Distinguishing marks",
}
_KIND_ALIASES = {"kind", "species", "type", "animal", "creature"}
_ROLE_ALIASES = {"role"}
_PRIORITY_ALIASES = {"priority", "standing", "billing"}
# mirrors services.channels.CAST_PRIORITIES, spelled out here for the same reason "main" /
# "supporting" is: this module imports nothing from the project so it stays trivially testable
_PRIORITIES = ("lead", "normal", "resting")
_PERSONALITY_ALIASES = {"personality", "traits", "character", "temperament", "nature"}


def _clean(text: str) -> str:
    """Drop inline markdown emphasis and collapse whitespace."""
    text = re.sub(r"(\*\*|__|`)", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _sentence(label: str, value: str) -> str:
    v = _clean(value).rstrip(".;,")
    return f"{label}: {v}." if v else ""


def _truncate_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(" ,;:-")


def _sections(md: str) -> list:
    """[(title, [lines])] split on level-2 headings; the text before the first `##` gets an
    empty title (it is the document title/preamble and is dropped by the caller)."""
    out, title, lines = [], "", []
    for raw in md.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _H2.match(raw)
        if m:
            out.append((title, lines))
            title, lines = _NUMBERED_TITLE.sub("", _clean(m.group(1))), []
        else:
            lines.append(raw)
    out.append((title, lines))
    return out


def _parse_characters(lines: list, max_kind_chars: int, max_name_chars: int, warnings: list) -> list:
    """Every `###` block of a section as a cast entry; blocks without a usable look are skipped."""
    blocks, cur = [], None
    for raw in lines:
        m = _H3.match(raw)
        if m:
            cur = (m.group(1), [])
            blocks.append(cur)
        elif cur is not None:
            cur[1].append(raw)
    out = []
    for heading, body in blocks:
        heading = _clean(heading)
        name, paren_kind = heading, ""
        pm = _PAREN_KIND.match(heading)
        if pm and pm.group(1).strip():
            name, paren_kind = pm.group(1).strip(), pm.group(2).strip()
        name = name.strip(" -:")
        if not name:
            warnings.append("a character block has no name - skipped")
            continue
        if len(name) > max_name_chars:
            warnings.append(f'name "{name[:20]}..." shortened to {max_name_chars} characters')
            name = name[:max_name_chars].rstrip()
        fields = []          # [label, value] in source order
        for raw in body:
            if not raw.strip():
                continue
            fm = _FIELD.match(raw)
            if fm:
                fields.append([fm.group(1).strip(), fm.group(2)])
            elif fields:      # continuation line of the previous field
                fields[-1][1] += " " + raw.strip()
        kind, role, personality = paren_kind, "main", ""
        priority = "normal"
        look_parts = {}
        extra = []
        for label, value in fields:
            key = label.lower().strip()
            if key in _KIND_ALIASES:
                kind = _clean(value)
            elif key in _ROLE_ALIASES:
                role = "main" if _clean(value).lower().startswith("main") else "supporting"
            elif key in _PRIORITY_ALIASES:
                # an unreadable value falls back to normal rather than failing the import: a bible
                # is a pasted document, and one odd word in it must not cost the whole roster
                word = _clean(value).lower()
                priority = next((p for p in _PRIORITIES if word.startswith(p)), "normal")
            elif key in _PERSONALITY_ALIASES:
                personality = _clean(value)
            elif key in _LOOK_ALIASES:
                canon = _LOOK_ALIASES[key]
                s = _sentence(canon, value)
                if s:
                    look_parts[canon] = (look_parts[canon] + " " + s) if canon in look_parts else s
            else:
                s = _sentence(label[:1].upper() + label[1:], value)
                if s:
                    extra.append(s)
        look = " ".join([look_parts[k] for k in LOOK_ORDER if k in look_parts] + extra)
        if not look:
            warnings.append(f"{name}: no look description found - skipped")
            continue
        if len(kind) > max_kind_chars:
            kind = _truncate_words(kind, max_kind_chars)
            warnings.append(f"{name}: kind shortened to {max_kind_chars} characters")
        out.append({"name": name, "kind": kind, "role": role, "look": look,
                    "personality": personality, "priority": priority})
    return out


def _rules_text(title: str, lines: list) -> str:
    body = []
    for raw in lines:
        if not raw.strip():
            continue
        m = _LIST_MARK.match(raw)
        marker = ""
        if m:
            marker = raw[m.start():m.end()].strip() + " "
            raw = raw[m.end():]
        line = _clean(raw)
        if line:
            body.append(marker + line)
    if not body:
        return ""
    return (f"{title}:\n" if title else "") + "\n".join(body)


def parse_bible(md: str, max_kind_chars: int = 60, max_name_chars: int = 40) -> dict:
    """{"rules": str, "cast": [ {name, kind, role, look, personality} ], "sections": [titles that
    became rules], "warnings": [str]} - never raises on odd input."""
    warnings, cast, rules, used_titles = [], [], [], []
    sections = _sections(md or "")
    has_h2 = len(sections) > 1
    for idx, (title, lines) in enumerate(sections):
        chars = _parse_characters(lines, max_kind_chars, max_name_chars, warnings) \
            if any(_H3.match(l) for l in lines) else []
        if chars:
            cast.extend(chars)
            continue
        if idx == 0 and has_h2:
            continue          # document title + preamble: the app has its own persona text
        if _FORMAT_SECTION.search(title):
            continue          # the app's own scene prompt format applies
        text = _rules_text(title, lines)
        if text:
            rules.append(text)
            used_titles.append(title or "(untitled)")
    seen = {}                 # duplicate names: the later definition wins
    for c in cast:
        key = c["name"].lower()
        if key in seen:
            warnings.append(f'{c["name"]} is defined twice - the later definition is used')
        seen[key] = c
    return {"rules": "\n\n".join(rules), "cast": list(seen.values()),
            "sections": used_titles, "warnings": warnings}
