"""The one thing a video promised its viewer.

A video that says "the exact prompt structure is in the description" has to hand that prompt over.
Once the narration is final, services.llm.write_resource decides whether the script makes such a
promise and writes the promised thing; this module puts it in the job folder in the three shapes a
viewer can use - resource.md (source), resource.txt (paste anywhere) and resource.pdf (a branded
download carrying the channel logo) - and folds it into the generated YouTube description.

The PDF is best effort. fpdf2 or a usable Unicode TTF may be missing, and a missing PDF must never
cost a job its video: to_pdf() returns a reason instead of raising, the caller logs it, and the .md
and .txt remain the guaranteed deliverable.
"""
import os
import re
import sys
from pathlib import Path

import config

MD_NAME, TXT_NAME, PDF_NAME = "resource.md", "resource.txt", "resource.pdf"
DIVIDER = "-" * 40
_CHROME = 200                # the block's own dividers, heading and blank lines, worst case
RESOURCE_MIN_CHARS = 1200    # never ask for less; a promised prompt is written in full either way

# DejaVu first, deliberately: it is the one face guaranteed in the container (fonts-dejavu-core,
# Dockerfile) and it is freely embeddable. Windows' Segoe UI comes first in the story board's search
# order but its embedding permissions are restricted for a PDF we hand to viewers.
_MONO_HINT = ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "consola.ttf", "cour.ttf")


# ---- markdown -> plain text ------------------------------------------------
def _inline(text: str) -> str:
    """Take the inline Markdown marks off, keep every word."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return re.sub(r"`([^`]+)`", r"\1", text)


def _bullet(line: str) -> str:
    """'- [ ] do it' -> '[ ] do it'; '* thing' -> '- thing'. Indentation is kept."""
    line = re.sub(r"^(\s*)[-*+]\s+\[([ xX])\]\s*",
                  lambda m: f"{m.group(1)}[{'x' if m.group(2).lower() == 'x' else ' '}] ", line)
    return re.sub(r"^(\s*)[-*+]\s+", r"\1- ", line)


def to_text(markdown: str) -> str:
    """The plain-text twin: the same words with the syntax removed.

    Fenced blocks are the part a viewer copies verbatim, so their content passes through byte for
    byte. Headings lose their '#' - which also matters for the YouTube description, where a stray
    "## Steps" would otherwise become the hashtag #Steps."""
    out, fence = [], False
    for line in (markdown or "").splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            out.append(line)
            continue
        head = re.match(r"\s*(#{1,6})\s+(.*)$", line)
        if head:
            text = _inline(head.group(2)).strip()
            out.append(text.upper() if len(head.group(1)) <= 2 else text)
        else:
            out.append(_inline(_bullet(line)).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def for_description(text: str) -> str:
    """The same text, made safe to paste into a YouTube description.

    Two platform rules bite here and neither is obvious:
      * '<' and '>' are rejected by the upload API - and a promised prompt is exactly the content
        full of <your topic> placeholders, so this fires constantly once a template is inlined;
      * YouTube reads the FIRST THREE hashtags in a description and ignores every hashtag when a
        description holds more than 15, so a '#role' or a '# comment' inside the resource would
        either outrank the channel's real tags or void the lot. The '#' is dropped, the word stays."""
    text = text.replace("<", "[").replace(">", "]")
    return re.sub(r"(?<!\w)#(?=\w)", "", text)


# ---- fonts -----------------------------------------------------------------
def _ttf(names: tuple, bold: bool = False) -> str:
    """First existing .ttf among `names`, searched where the story board already looks."""
    from services import storyboard      # lazy: storyboard pulls PIL, llm and video
    roots, seen = [], set()
    for cand in storyboard._font_candidates(bold):
        try:
            parent = str(Path(cand).parent)
        except Exception:  # noqa: BLE001
            continue
        if parent and parent not in seen:
            seen.add(parent)
            roots.append(parent)
    for name in names:
        for root in roots:
            p = Path(root) / name
            if p.is_file():
                return str(p)
    return ""


def _pdf_font(bold: bool) -> str:
    """A real TTF for fpdf2: the core PDF fonts are latin-1 only, and a promised prompt may contain
    any character, so an embedded Unicode face is not optional here."""
    preferred = (("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf")
                 if bold else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"))
    hit = _ttf(preferred, bold)
    if hit:
        return hit
    from services import storyboard
    for cand in storyboard._font_candidates(bold):   # last resort: whatever is actually installed
        try:
            p = Path(cand)
            if cand and p.suffix.lower() == ".ttf" and p.is_file():
                return str(p)
        except Exception:  # noqa: BLE001
            continue
    return ""


def pdf_available() -> bool:
    """True when a resource.pdf can actually be produced here."""
    try:
        import fpdf
        return (int(str(getattr(fpdf, "FPDF_VERSION", "0")).split(".")[0]) >= 2
                and bool(_pdf_font(False)))
    except Exception:  # noqa: BLE001
        return False


# ---- PDF -------------------------------------------------------------------
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\U00002190-\U000021FF]")


def _hard_wrap(text: str, width: int = 110) -> str:
    """Break runs longer than `width` characters. fpdf2 raises when a single unbreakable token is
    wider than the text column, and a pasted prompt, path or URL is exactly that."""
    out, cur = [], []
    for word in text.split(" "):
        while len(word) > width:
            cur.append(word[:width])
            out.append(" ".join(cur))
            cur, word = [], word[width:]
        cur.append(word)
    out.append(" ".join(cur))
    return "\n".join(out)


def _logo_png(logo, tmp_dir: Path):
    """The channel logo flattened onto white. The stored logos are RGBA and a PDF page has no
    transparency, so compositing here keeps fpdf2 from drawing a black box."""
    try:
        from PIL import Image
        if not logo or not Path(logo).exists():
            return None
        img = Image.open(logo).convert("RGBA")
        img.thumbnail((240, 240), Image.LANCZOS)
        flat = Image.new("RGB", img.size, "white")
        flat.paste(img, (0, 0), img)
        out = tmp_dir / "_pdf_logo.png"
        flat.save(out, "PNG")
        return out
    except Exception:  # noqa: BLE001 - the header simply loses its logo
        return None


def to_pdf(markdown: str, path: Path, title: str = "", channel: str = "", video_title: str = "",
           logo=None) -> str:
    """Render the resource as a branded A4 PDF: channel logo and name in the header, the resource
    title and the video it came from, then the body, with a page footer.

    Returns "" on success, or a short reason why no file was written - the PDF is a convenience,
    never a reason to fail a job."""
    try:
        import fpdf
        from fpdf import FPDF
    except Exception as e:  # noqa: BLE001
        return f"fpdf2 is not installed ({e.__class__.__name__}) - pip install -r requirements.txt"
    if int(str(getattr(fpdf, "FPDF_VERSION", "0")).split(".")[0]) < 2:
        return "the installed 'fpdf' is PyFPDF, not fpdf2 (pip install fpdf2)"
    regular = _pdf_font(False)
    if not regular:
        return "no Unicode TTF found - install fonts-dejavu-core or set VIDEOGEN_FONT_REGULAR"
    bold = _pdf_font(True) or regular
    mono = _ttf(_MONO_HINT) or regular
    path = Path(path)
    # DejaVu has no emoji and the colour-emoji faces are bitmap fonts fpdf2 cannot embed, so an
    # emoji would render as a .notdef box. The .md and .txt keep them.
    body_md = _EMOJI.sub("", markdown or "")
    logo_png = _logo_png(logo, path.parent)

    class Doc(FPDF):
        def header(self):
            if self.page_no() == 1 and logo_png:
                try:
                    self.image(str(logo_png), x=20, y=10, h=11)
                except Exception:  # noqa: BLE001
                    pass
            self.set_y(11)
            self.set_font("body", "B", 9)
            self.set_text_color(110, 110, 118)
            self.cell(0, 6, channel or "", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(220, 220, 226)
            self.line(20, 23, self.w - 20, 23)
            self.set_text_color(20, 20, 24)
            self.set_y(30)

        def footer(self):
            self.set_y(-14)
            self.set_font("body", "", 8)
            self.set_text_color(130, 130, 138)
            left = (video_title or "")[:70]
            self.cell(0, 5, left, align="L")
            self.set_y(-14)
            self.cell(0, 5, f"page {self.page_no()} of {{nb}}", align="R",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(20, 20, 24)

    pdf = Doc(format="A4", unit="mm")
    pdf.set_margins(20, 26, 20)
    pdf.set_auto_page_break(True, margin=20)
    for family, style, src in (("body", "", regular), ("body", "B", bold), ("mono", "", mono)):
        try:
            pdf.add_font(family, style, src)
        except TypeError:                       # a very old build still wants uni=
            pdf.add_font(family, style, src, uni=True)
    if title:
        pdf.set_title(title)
    if channel:
        pdf.set_author(channel)
    pdf.add_page()

    if title:                                   # title block
        pdf.set_font("body", "B", 19)
        pdf.multi_cell(0, 9, _hard_wrap(title), align="L", new_x="LMARGIN", new_y="NEXT")
    if video_title:
        pdf.set_font("body", "", 10)
        pdf.set_text_color(110, 110, 118)
        pdf.multi_cell(0, 5.5, _hard_wrap(f"From the video: {video_title}"),
                       align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(20, 20, 24)
    pdf.ln(4)

    fence, block, skip_h1 = False, [], bool(title)
    for raw in body_md.splitlines():
        if skip_h1 and raw.lstrip().startswith("# "):
            skip_h1 = False           # the title block above already carries it
            continue
        if raw.strip():
            skip_h1 = False
        if raw.lstrip().startswith("```"):
            if fence:
                _pdf_code(pdf, block)
                block = []
            fence = not fence
            continue
        if fence:
            block.append(raw)
        else:
            _pdf_line(pdf, raw)
    if block:
        _pdf_code(pdf, block)
    pdf.output(str(path))
    if logo_png:
        Path(logo_png).unlink(missing_ok=True)
    return ""


def _pdf_line(pdf, raw: str):
    head = re.match(r"\s*(#{1,6})\s+(.*)$", raw)
    if head:
        level = len(head.group(1))
        pdf.ln(3 if level > 1 else 0.5)
        pdf.set_font("body", "B", 15 if level == 1 else 12.5 if level == 2 else 11)
        pdf.multi_cell(0, 6.6, _hard_wrap(_inline(head.group(2)).strip()),
                       align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)
        return
    text = _inline(_bullet(raw)).rstrip()
    if not text.strip():
        pdf.ln(2.5)
        return
    pdf.set_font("body", "", 10.5)
    pdf.multi_cell(0, 5.2, _hard_wrap(text), align="L", new_x="LMARGIN", new_y="NEXT")


def _pdf_code(pdf, lines: list):
    """A fenced block - the part a viewer copies verbatim - in mono on a tinted background,
    hard-wrapped so a 400-character prompt line cannot overflow the column."""
    body = "\n".join(_hard_wrap(ln, 84) for ln in lines).strip("\n")
    if not body:
        return
    pdf.ln(1)
    pdf.set_font("mono", "", 8.8)
    pdf.set_fill_color(244, 244, 246)
    pdf.multi_cell(0, 4.4, body, fill=True, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("body", "", 10.5)
    pdf.ln(2)


# ---- files + description ---------------------------------------------------
def write_files(job_dir: Path, res: dict, channel: str = "", video_title: str = "",
                logo=None) -> tuple:
    """resource.md / .txt / .pdf in the job folder. Returns (names written, pdf warning)."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    md = (res.get("markdown") or "").strip() + "\n"
    (job_dir / MD_NAME).write_text(md, encoding="utf-8")
    (job_dir / TXT_NAME).write_text(to_text(md), encoding="utf-8")
    names, warn = [MD_NAME, TXT_NAME], ""
    try:
        warn = to_pdf(md, job_dir / PDF_NAME, title=res.get("title", ""), channel=channel,
                      video_title=video_title, logo=logo)
    except Exception as e:  # noqa: BLE001 - a PDF is never worth a video
        warn = f"{e.__class__.__name__}: {e}"
    if warn:
        (job_dir / PDF_NAME).unlink(missing_ok=True)   # never leave half a PDF behind
    else:
        names.append(PDF_NAME)
    return names, warn


def clear(job_dir: Path):
    """Drop any resource files left by an earlier run - the writer owns them, so a job that now
    promises nothing must not keep a hand-out describing a narration it no longer has."""
    for name in (MD_NAME, TXT_NAME, PDF_NAME):
        (Path(job_dir) / name).unlink(missing_ok=True)


def markdown_budget(description: str, limit: int = None) -> int:
    """The character cap handed to the writer: what is left of the YouTube description after the
    SEO text and the block's own chrome - never below RESOURCE_MIN_CHARS, because a promised prompt
    is written in full even when it then has to live in the files only."""
    limit = config.YT_DESCRIPTION_LIMIT if limit is None else limit
    return max(RESOURCE_MIN_CHARS, limit - len((description or "").rstrip()) - _CHROME)


def description_block(res: dict) -> str:
    """The block appended to the YouTube description: the promised thing itself, as plain text,
    under a heading someone skimming the description will find."""
    head = (res.get("title") or f"the {res.get('kind') or 'resource'} from this video").strip()
    body = for_description(to_text(res.get("markdown", "")).strip())
    return f"{DIVIDER}\n{for_description(head.upper())}\n{DIVIDER}\n\n{body}"


def fold_into_description(plan: dict, limit: int = None) -> bool:
    """Append the resource to the generated YouTube description, once, in place.

    plan["seo_description"] stays the single publish-ready text everything already reads (the
    upload, the ZIP's <Title>.txt, the job page's SEO card), and the writer's untouched version is
    kept as "seo_description_base" so a re-run can never append the block twice.

    The block goes at the very END, after the hashtag line write_seo puts there: YouTube shows the
    FIRST THREE hashtags of a description above the title, so anything inserted above that line
    would let a stray '#' in the resource outrank the channel's real tags."""
    limit = config.YT_DESCRIPTION_LIMIT if limit is None else limit
    res = plan.get("resource") or {}
    desc = (plan.get("seo_description") or "").rstrip()
    if not res.get("offered") or not desc or plan.get("seo_description_base") is not None:
        return False
    joined = f"{desc}\n\n{description_block(res)}"
    if len(joined) > limit:
        res["in_description"] = False
        res["description_over_by"] = len(joined) - limit
        return False
    plan["seo_description_base"] = desc
    plan["seo_description"] = joined
    res["in_description"] = True
    res.pop("description_over_by", None)
    return True


def attach(job_dir: Path, plan: dict, res: dict, channel: str = "", logo=None) -> tuple:
    """Everything that follows the writer's answer: the plan.json record, the files in the job
    folder and the description append. Returns (record, log lines)."""
    if not res:
        clear(job_dir)
        return ({"offered": False, "checked": True},
                ["the script promises the viewer nothing - no resource generated"])
    video_title = plan.get("seo_title") or plan.get("title") or ""
    names, warn = write_files(job_dir, res, channel=channel, video_title=video_title, logo=logo)
    record = {"offered": True, "kind": res.get("kind", ""), "title": res.get("title", ""),
              "promise": res.get("promise", ""), "markdown": res.get("markdown", ""),
              "files": names, "chars": len(res.get("markdown", "")),
              "in_description": False, "pdf_error": warn}
    plan["resource"] = record            # fold_into_description reads it back off the plan
    logs = [f'resource: the script promises a {record["kind"] or "resource"} - wrote '
            f'{", ".join(names)} ({len(record["markdown"].split())} words)']
    if warn:
        logs.append(f"resource.pdf skipped ({warn}); the markdown and text versions are there")
    if fold_into_description(plan):
        logs.append(f"resource appended to the description "
                    f"({len(plan['seo_description'])}/{config.YT_DESCRIPTION_LIMIT} characters used)")
    elif plan.get("seo_description"):
        logs.append(f"resource left out of the description: the block is "
                    f"{record.get('description_over_by', 0)} characters over the "
                    f"{config.YT_DESCRIPTION_LIMIT}-character limit - paste it from resource.txt "
                    f"when you upload")
    return record, logs
