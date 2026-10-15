"""vLLM (OpenAI-compatible) planning: narration first, then one scene prompt per narration segment.

Two-step design: the narration is written and voiced first; only then — once the real clip count
is known from the audio duration — are the scene prompts generated, one per narration segment,
so the visuals stay in sync with what is being said all the way to the end.
"""
import json
import re
import sys

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

import config
from services import cast, lyrics

AUDIENCE_RULES = {
    "kids": ("All content must be family-friendly, gentle and safe for young kids: no violence, "
             "gore, fear, romance, innuendo or mature themes - even dark or spooky styles stay "
             "playful and reassuring. "),
    "family": ("All content must suit a general family audience: no graphic violence or gore, no "
               "explicit sexual content, no profanity; mild peril, humor and light romance are fine. "),
    "general": ("Content must follow YouTube community and advertiser guidelines: no explicit "
                "sexual content, no graphic gore, no hate or harassment; otherwise mature and "
                "dramatic themes are allowed. "),
}


def _channel(ch: dict | None) -> dict:
    """Normalize a job's channel snapshot, filling in whatever a job created before
    channels existed is missing. An unknown audience falls back to `general`, the same
    default a newly created channel gets."""
    ch = dict(ch or {})
    ch.setdefault("name", config.CHANNEL_NAME)
    ch.setdefault("purpose", "")
    ch.setdefault("rules", "")
    ch.setdefault("cast", [])
    if ch.get("audience") not in AUDIENCE_RULES:
        ch["audience"] = "general"
    if ch.get("video_type") not in config.VIDEO_TYPES:
        ch["video_type"] = "animated"
    return ch


VIDEO_KIND = {
    "animated": "animated",
    "real": "realistic, live-action-style",
    "tech": "realistic, live-action-style technology explainer",
}

TECH_RULES = ("This is a TECH channel: be concrete and technically accurate - name the actual "
              "technologies, mechanisms, numbers and trade-offs instead of vague hype; explain like a "
              "knowledgeable, friendly tech presenter. Visuals favour devices, screens and interfaces, "
              "code, data centres, chips, robots, dashboards and clean abstract data visualisations. ")


SPOKEN_OVERRIDE = ("The channel rules and the character sheet above describe how the characters are DRAWN: "
                   "they apply to image and video prompts only. What you write here is SPOKEN ALOUD, so it "
                   "must never contain appearance detail - name a character and move on. ")

BRIEF_OVERRIDE = ("The channel rules and the character sheet above describe how the characters are DRAWN: "
                  "they apply to the finished image and video prompts, and each character's full look is "
                  "attached to those automatically, later. What you write here is a short PLAN, not an "
                  "image prompt - name a character and say what they do, never repeat or invent appearance "
                  "detail, and never write camera, lens, shot or art-style words. ")

# ---- channels whose videos have no characters (cast.enabled() is False) ----
# The cast machinery goes quiet on its own (services/cast.py active() returns []), but an empty
# roster is not an instruction: with nothing said, the writer happily invents a protagonist. These
# carry the positive brief instead.
NO_CHARACTER_RULES = (
    "THIS CHANNEL HAS NO CHARACTERS. Never invent a protagonist, a named person, a host or a mascot, "
    "and never tell a story about someone - no 'Alex, a project manager drowning in email', no arc, "
    "no cast. The video speaks straight to the viewer in the second person ('you', 'your'). Build "
    "every video as HOOK (name the viewer's problem or the promise in one line) -> PROBLEM (why it "
    "costs them time, money or nerves, concretely) -> STEPS (the actual specific things to do, or "
    "the demo, in order, each one usable on its own) -> PAYOFF (what they can do now, and the one "
    "next step). Generic unnamed people MAY appear ON SCREEN as illustration - a project manager at "
    "a laptop, hands typing, a presenter gesturing at a screen - described only by role, rough age "
    "and clothing. They are never named, never introduced, never given a backstory, never recur, "
    "and the spoken words never refer to them. If the channel's style & rules text above carries "
    "character rules, they describe a cast this channel does not use: ignore every instruction "
    "there about characters, their looks or their consistency, and keep only its visual style, tone "
    "and format guidance. ")

SPOKEN_NO_CHARACTER_OVERRIDE = (
    "The channel rules above describe how the video LOOKS: they apply to image and video prompts "
    "only. What you write here is SPOKEN ALOUD, so it must never carry appearance detail and must "
    "never describe the people shown on screen - the visuals carry that. ")

NO_CHARACTER_SCENE_RULE = (
    "\nPEOPLE: this video has no characters. The subject of every shot is the topic itself - the "
    "screens, tools, objects, places and steps being shown. Generic unnamed people MAY appear where "
    "they make a beat clearer: a project manager at a laptop, hands typing, a presenter gesturing "
    "at a screen, an over-the-shoulder view of someone working. Describe them by role, rough age "
    "and clothing only - never name them, never give them a backstory or a reaction arc, never "
    "write dialogue for them, and never build the video around one recurring person. Continuity "
    "between clips is carried by the setting, the screen content and the camera move, not by a "
    "person.")

NARRATION_CHARACTER_RULE = "Refer to any character by name only and never describe how they look."
NARRATION_EXPLAINER_RULE = (
    "Write it as an informative explainer spoken straight to the viewer in the second person "
    "('you', 'your'): open on the hook, name the problem in their terms, walk through the concrete "
    "steps or the demo in order - real actions, real settings, real numbers, one step at a time - "
    "and close on the payoff and the one next step. No protagonist, no named person, no story, no "
    "dialogue, no 'meet Alex': where an example helps keep it about 'you' or an unnamed role ('a "
    "project manager', 'your team') and never give it a name or a backstory.")


def _narration_rule(channel: dict | None) -> str:
    """The clause keeping spoken words free of appearance detail (cast channels) or free of
    invented people altogether (explainer channels)."""
    return NARRATION_CHARACTER_RULE if cast.enabled(channel) else NARRATION_EXPLAINER_RULE


def _system(channel: dict | None = None, used: list | None = None, rules: bool = True,
            spoken: bool = False, brief: bool = False) -> str:
    """The per-job system prompt.

    used:  None while the script is being written (the cast sheet lists the whole roster and asks
           the writer to pick); afterwards the video's cast_used names, so every later call only
           carries the looks of the characters that are actually in this video.
    rules: include the channel's fixed style & rules text (off for calls that only need names,
           such as SEO and music).
    spoken: the call writes words a voice reads out (narration, script review, song lyrics). The
           cast sheet then carries no looks, and the override cancels the "repeat the look"
           instructions a channel's own rules text may contain.
    brief: the call writes a short plan that BECOMES an image prompt later (story beats). The sheet
           carries no looks either - the look is re-attached deterministically by cast.with_prefix
           and by write_scenes' own full sheet - but the override says why, so the writer does not
           simply forget the characters exist."""
    ch = _channel(channel)
    chars = cast.enabled(ch)
    kind = VIDEO_KIND.get(ch["video_type"], VIDEO_KIND["real"])
    purpose = (f'Channel purpose: "{ch["purpose"]}" - keep every video aligned with this purpose '
               "and tone. ") if ch["purpose"].strip() else ""
    rules_block = (f"CHANNEL STYLE & RULES (fixed reference for every story and scene):\n"
                   f"{ch['rules'].strip()}\n") if rules and ch["rules"].strip() else ""
    opening = (f"You are a creative director turning source content into a short, original "
               f"AI-generated {kind} video. ") if chars else (
               f"You are an expert explainer turning source content into a short, original "
               f"AI-generated {kind} video that teaches the viewer something they can use "
               f"themselves. ")
    return (opening
            + AUDIENCE_RULES[ch["audience"]]
            + (TECH_RULES if ch["video_type"] == "tech" else "")
            + purpose
            + f'The videos are published on the YouTube channel "{ch["name"]}": if the '
            "source content mentions any other channel, brand, show or creator name, NEVER use it - "
            f'use "{ch["name"]}" instead wherever a name is called for, and mention the '
            "channel naturally only where it fits (intros, outros, sign-offs), without forcing it "
            "into every line. "
            + rules_block
            + (cast.sheet_text(ch, used=used, spoken=spoken or brief) if chars else NO_CHARACTER_RULES)
            + (BRIEF_OVERRIDE if (brief and chars) else
               (SPOKEN_OVERRIDE if chars else SPOKEN_NO_CHARACTER_OVERRIDE) if (spoken or brief) else "")
            + "Always answer with a single JSON object and nothing else.")


def _cast_selection_rule(channel: dict | None, request_text: str = "", verbatim: bool = False) -> str:
    """Script-writing instruction for choosing this video's characters from the channel roster.
    Names the user wrote in the request (topic/description) are mandatory. verbatim=True is the
    pasted-transcript case: the words are fixed, the choice only decides who plays in the visuals."""
    if not cast.active(channel):
        return ""
    forced = cast.names(cast.used_characters(channel, request_text))
    if verbatim:
        rule = ("\nCAST SELECTION: the script above is fixed, but decide which of the channel's recurring "
                "characters (if any) play in the visuals of this video - only the ones this story needs; "
                "every other cast member stays out.")
    else:
        rule = ("\nCAST SELECTION: choose from the channel roster only the characters this story needs "
                "(usually 1-3; MAIN characters are the usual protagonists, supporting characters when the "
                "story calls for them) and refer to them by name in the narration; every other cast "
                "member stays out entirely.")
    if forced:
        rule += (f" The request names {', '.join(forced)}: these exactly must be used, in the roles the "
                 "request gives them (e.g. main / supporting).")
    return rule + ' Also return "characters": the list of cast names used in this video (may be empty). '


def _faithful_keys(channel: dict | None) -> str:
    """What a pasted description must be kept faithful to; an explainer channel has no characters
    to preserve, and naming them here invites the writer to invent some."""
    return "content, characters, structure and intent" if cast.enabled(channel)         else "content, structure and intent"


def _chosen(channel: dict | None, value) -> list:
    """The cast names the LLM reported under "characters", validated against the roster."""
    if isinstance(value, str):
        value = re.split(r"[,;/]| and ", value)
    if not isinstance(value, list):
        return []
    return cast.names(cast.select_characters(channel, [str(v) for v in value]))

STYLE_EXAMPLE = ("Realistic live-action cinematic look, practical film photography style, anamorphic lens, "
                 "shallow depth of field, film grain, restrained grading for a premium feel, natural movement.")
TECH_STYLE_EXAMPLE = ("Sleek modern tech-documentary look: real studios, labs and offices, devices and glowing "
                      "screens, cool blue-and-graphite palette with warm accent lights, crisp shallow depth of "
                      "field, smooth gimbal and slider camera moves, premium product-film polish.")


def _style_example(channel: dict | None) -> str:
    return TECH_STYLE_EXAMPLE if _channel(channel)["video_type"] == "tech" else STYLE_EXAMPLE


def _style_rule(art_style: str, channel: dict | None) -> str:
    """What the writer is asked to return under "style": the one sentence that becomes the global
    look of every shot, pinned to the job's art style when it has one."""
    if art_style:
        return (f'one sentence describing the global visual style applied to every shot. '
                f'It MUST follow this art direction: "{art_style}" - keep that exact aesthetic '
                f'and expand it with detail that fits this video.')
    return (f'one sentence describing the global visual style applied to every shot, similar to: '
            f'"{_style_example(channel)}"')

SCENE_RULES = """Each scene prompt describes ONE continuous {secs}-second shot for a text-to-video model and must contain, in this order:
- Scene overview: subject, setting, action, mood (2-3 sentences).
- Storyboard with timestamps, e.g. "[0s-2s] ... [2s-4s] ... [4s-{secs}s] ...". When a TIMELINE of what is
  heard is given for the clip, the storyboard timestamps follow it: each visual beat starts when its
  line is heard.
- Camera: angle and movement.
- Audio: ambient sound / effects appropriate to the scene (no speech, no narration, no music with lyrics).
- End with: "No text, subtitles, logos or watermarks of any kind."
CONTINUITY CONTRACT (the video model renders each clip starting from the LAST FRAME of the previous clip):
- The [0s] beat of a scene IS the final frame of the previous scene: same camera angle, same characters in
  the same positions, same setting and lighting. Never open with a cut, a new shot, a different angle, or a
  teleport to another place.
- Reframe only through continuous on-screen camera motion (pan / dolly / zoom) WITHIN the clip; introduce
  new elements by having them enter the frame or be revealed by that motion.
- End every scene on a stable, held end-state (subjects clearly in frame, motion settling) - never mid
  whip-pan, mid-blur or mid-cut - because the next clip is seeded from that exact ending frame.
Never mention real people's names or brands."""


def _client():
    """The vLLM client, with the timeouts spelled out. The SDK default is a 600-second read
    with two silent retries, so a server that accepts a request and then never decodes a token
    - a wedged engine answers /v1/models instantly all the while - held a job at "writing script
    with the LLM" for up to an hour per call with nothing written to the log. Retries are off for
    the same reason: re-sending a request that timed out only pays the wait again, and a failed
    job can be resumed from the step it stopped at. The connect timeout stays short so an
    unreachable box fails in seconds."""
    return OpenAI(base_url=config.VLLM_URL, api_key="none", max_retries=0,
                  timeout=httpx.Timeout(float(config.LLM_TIMEOUT), connect=10.0))


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def _chat(messages, max_tokens=8000) -> dict:
    c = _client()
    kwargs = dict(model=config.VLLM_MODEL, messages=messages, temperature=0.7, max_tokens=max_tokens,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    try:
        try:
            r = c.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except APIStatusError:   # answered and refused: this server may not support JSON mode
            r = c.chat.completions.create(**kwargs)
    except APITimeoutError as e:   # subclass of APIConnectionError, so it is caught first
        raise RuntimeError(
            f"the LLM at {config.VLLM_URL} accepted the request but did not answer within "
            f"{config.LLM_TIMEOUT}s - check that vLLM is still serving {config.VLLM_MODEL} "
            f"(restart it if it is wedged), then resume this job") from e
    except APIConnectionError as e:
        raise RuntimeError(f"could not reach the LLM at {config.VLLM_URL} ({e}) - start vLLM "
                           f"or fix the URL in Settings, then resume this job") from e
    msg = r.choices[0].message
    content = msg.content or getattr(msg, "reasoning", None) or ""
    if not content.strip():
        raise RuntimeError(f"vLLM returned no content (finish_reason={r.choices[0].finish_reason})")
    return _parse_json(content)


def _as_strings(scenes) -> list:
    return [s if isinstance(s, str) else s.get("prompt", json.dumps(s)) for s in scenes]


def write_narration(transcript: str, words: int | None = None, art_style: str = "",
                    channel: dict | None = None) -> dict:
    """Step 1: title, narration script and global visual style (no scenes yet)."""
    words = words or config.NARRATION_WORDS
    transcript = transcript[: config.MAX_TRANSCRIPT_CHARS]
    if art_style:
        style_rule = (f'one sentence describing the global visual style applied to every shot. '
                      f'It MUST follow this art direction: "{art_style}" - keep that exact aesthetic '
                      f'and expand it with detail that fits this video\'s subject.')
    else:
        style_rule = f'one sentence describing the global visual style applied to every shot, similar to: "{_style_example(channel)}"'
    user = f"""Transcript of the source video:
\"\"\"{transcript}\"\"\"

Create a new short video inspired by this content.{_cast_selection_rule(channel)} Return JSON with keys:
- "title": short title.
- "narration": a voiceover script of about {words} words, plain spoken English, no headings, no stage directions, no emojis. It will be read aloud by a TTS voice. {_narration_rule(channel)} Stay close to {words} words.
- "style": {style_rule}
"""
    plan = _chat([{"role": "system", "content": _system(channel, spoken=True)}, {"role": "user", "content": user}])
    for key in ("title", "narration", "style"):
        plan.setdefault(key, "")
    plan["cast_chosen"] = _chosen(channel, plan.pop("characters", None))
    if not plan["narration"].strip():
        raise RuntimeError("vLLM returned an empty narration")
    if config.SCRIPT_REVIEW:
        plan["narration"], plan["review_issues"] = review_script(
            plan["narration"], f'a video titled "{plan.get("title", "")}"', channel=channel)
    return plan


def write_script_from_prompt(text: str, source_kind: str, art_style: str = "",
                             words: int | None = None, feedback: str = "",
                             previous: str | None = None,
                             channel: dict | None = None, hook: str = "") -> dict:
    """Draft a video script (title/narration/style) from a user topic, description or pasted transcript.

    hook: the opening line a weekly plan chose for this video (services/planner.py); the script
    starts from it. Empty for every other job."""
    words = words or config.NARRATION_WORDS
    style_rule = _style_rule(art_style, channel)
    narration_rule = (f'the full script as a voiceover of about {words} words, plain spoken English, '
                      f'no headings, no stage directions, no emojis. It will drive the whole video. '
                      f'{_narration_rule(channel)} '
                      f'Stay close to {words} words.')
    if source_kind == "transcript":
        text = text[: config.MAX_TRANSCRIPT_CHARS]
        task = ('The user pasted the COMPLETE script for a video below - a transcript or song lyrics. '
                'Use it AS the narration: keep their wording, order, content and length. Do NOT expand, '
                'shorten, summarise, reorder or rewrite it. Only fix real problems: typos and grammar, '
                'timestamps, cue tags like [Music], speaker labels, section tags like [Verse] or [Chorus], '
                'and anything a TTS voice could not read aloud naturally. If the text has no problems, '
                'return it unchanged:\n"""' + text + '"""')
        narration_rule = ('the corrected script, plain spoken English, no headings, no stage directions, '
                          'no emojis. It will be read aloud by a TTS voice and drive the whole video.')
    elif source_kind == "description":
        task = ('The user wrote the following description / short script for a video. Build the video '
                f'script from it, staying FAITHFUL to their {_faithful_keys(channel)} - '
                'expand and polish it, do not replace it:\n"""' + text + '"""')
    elif source_kind == "repo":
        text = text[: config.MAX_TRANSCRIPT_CHARS]
        task = ('Below are the facts and the README / model card of a real open-source project - a '
                'GitHub repository or a Hugging Face model, dataset or space. Write an INFORMATIVE '
                'explainer video about THIS project for a viewer who has never seen it. Everything '
                'you say must come from the text below: never invent features, numbers, benchmarks, '
                'licences, authors or release dates that are not there, and never guess what the '
                'project does from its name. Cover, in this order and only as far as the text '
                'supports it: what the project is and the ONE problem it solves; who it is for, and '
                'when to reach for it instead of the obvious alternative; how it actually works - '
                'the real API, command, model size, architecture or training data the README names; '
                'what it takes to run it (install, hardware, dependencies, licence); and the one '
                'concrete thing the viewer should do with it first. Name the project by its real '
                'name. Use the numbers the facts give (stars, downloads, likes, parameter count, '
                'dataset size) once each, where they mean something - never read the fact list out. '
                'Skip the README furniture: badges, tables of contents, contribution guidelines, '
                'codes of conduct, sponsor sections and changelogs are not the story. Turn code into '
                'plain spoken English - say "one pip install and three lines of Python", never '
                'dictate a snippet, a flag or a file path character by character, because a TTS '
                'voice reads this aloud:\n"""' + text + '"""')
    elif cast.enabled(channel):
        task = 'Create an original video script for this channel about this topic: "' + text + '"'
    else:
        task = ('Create an original INFORMATIVE video script for this channel that teaches this topic '
                'to someone who is not technical, step by step: "' + text + '"')
    # a pasted transcript is kept verbatim; the selection then only decides who plays in the visuals
    task += _cast_selection_rule(channel, text[:600] if source_kind == "repo" else text,
                                 verbatim=source_kind == "transcript")
    if hook.strip() and source_kind != "transcript":
        task += ('\nThe video opens on this hook - use it as the first spoken line, word for word: '
                 f'"{hook.strip()}"')
    revision = ""
    if feedback:
        revision = '\n\nThe user reviewed a previous draft and asked for these changes: "' + feedback + '"'
        if previous:
            revision += '\nPrevious draft narration:\n"""' + previous + '"""\nRevise it accordingly.'
    user = f"""{task}{revision}

Return JSON with keys:
- "title": short title.
- "narration": {narration_rule}
- "style": {style_rule}
"""
    plan = _chat([{"role": "system", "content": _system(channel, spoken=True)}, {"role": "user", "content": user}])
    for key in ("title", "narration", "style"):
        plan.setdefault(key, "")
    plan["cast_chosen"] = _chosen(channel, plan.pop("characters", None))
    if not plan["narration"].strip():
        raise RuntimeError("vLLM returned an empty script")
    if config.SCRIPT_REVIEW:
        plan["narration"], plan["review_issues"] = review_script(
            plan["narration"], f'a video about: "{text[:300]}"', channel=channel,
            verbatim=source_kind == "transcript")
    return plan


SHORT_SCRIPT_RATIO = 0.85   # a draft below target * this is extended once before it is spoken


def extend_narration(narration: str, words: int, channel: dict | None = None) -> str:
    """Rewrite a too-short script up to its word target, keeping the same story and ending.

    The word target comes from the job's video length; a draft that falls well under it would
    make a video shorter than the user asked for. Called before any TTS and before the user
    approves the script, so nothing approved is ever rewritten."""
    have = len(narration.split())
    user = f"""This voiceover script is TOO SHORT: {have} words where it must be about {words} words.

\"\"\"{narration}\"\"\"

Rewrite it LONGER, to about {words} words. Keep the same story, the same order and the same ending -
expand what is already there: more detail on each beat, more examples, more explanation. Do not add a
new topic, do not repeat yourself, and do not pad with filler. It will be read aloud by a TTS voice,
so keep it plain spoken English with no headings, no stage directions and no emojis.
{_narration_rule(channel)}
Return JSON with one key:
- "narration": the longer script.
"""
    out = _chat([{"role": "system", "content": _system(channel, spoken=True)},
                 {"role": "user", "content": user}])
    longer = str(out.get("narration") or "").strip()
    return longer if len(longer.split()) > have else narration


def split_narration(narration: str, n: int) -> list:
    """Split the narration into n roughly equal word segments."""
    words = narration.split()
    if n <= 1:
        return [narration]
    per = len(words) / n
    segments, start = [], 0
    for i in range(n):
        end = len(words) if i == n - 1 else round((i + 1) * per)
        segments.append(" ".join(words[start:end]))
        start = end
    return segments


def _secs(x) -> str:
    """Seconds for prompt text without float noise: 5.2, 2.3, 5."""
    return f"{float(x):.1f}".rstrip("0").rstrip(".")


# ---- story videos ----------------------------------------------------------
# Two calls, in this order, and the order is the whole point. The BEATS are the video content -
# what happens, and therefore what is drawn. The LINES are the voiceover, written FROM those beats.
# Doing it the other way round is exactly what produces narration that describes the picture
# ("Ellie the elephant, wearing a yellow dress, is riding a blue train"), because the writer is then
# describing a shot it has only just imagined. Here the picture already exists when the words are
# written, so the words are free to simply tell the story.

BEAT_RULES = """BEAT RULES:
- The beats are a STORY, not a list of moments: beat 1 sets things up, the middle beats raise and
  answer one small problem, and the last beat ends it. Nothing is left hanging.
- ONE action per beat. If a beat needs the words "and then", it is two beats.
- NEVER describe how anyone or anything LOOKS - no colours, clothes, fur, feathers, size, materials,
  markings or ages. Name the character and say what they do; their look is added later, automatically.
- NEVER write camera, lens, shot, lighting, editing or art-style words (close-up, pan, cut, wide
  shot, zoom, cinematic, 4K). The shot is written later, from the beat.
- No dialogue here and no quotation marks: the spoken words are written separately, from these beats."""


def write_story_beats(text: str, source_kind: str, n: int, clip_seconds: float,
                      art_style: str = "", feedback: str = "", previous: dict | None = None,
                      channel: dict | None = None, hook: str = "", model: str = "") -> dict:
    """The VIDEO content of a story video: title, premise, visual style and exactly n story beats.

    One call, not one per beat: the beats have to work as a single story, unlike write_scenes, where
    one call per clip is what stops a scene drifting from the fixed audio segment it must match.

    Written with the cast sheet in `brief` mode, so the beats carry no appearance detail at all -
    the canonical look is re-attached deterministically by cast.with_prefix at render time and by
    write_scenes' own full sheet. Returns {title, premise, style, beats, cast_chosen}, where each
    beat is {"setting", "action", "characters"}.

    model: the job's video model, which decides what a requested clip length really renders as."""
    secs = _secs(config.actual_clip_seconds(clip_seconds, model))
    if source_kind == "description":
        task = ('The user wrote the following description / outline for a story video. Build the '
                f'story from it, staying FAITHFUL to their {_faithful_keys(channel)} - expand and '
                'polish it, do not replace it:\n"""' + text + '"""')
    elif source_kind in ("topic", ""):
        task = 'Create an original STORY for this channel about this topic: "' + text + '"'
    else:
        task = ('Below is source content. Turn it into an original STORY for this channel that '
                'carries the same ideas, told as something that happens to the characters:\n"""'
                + text[: config.MAX_TRANSCRIPT_CHARS] + '"""')
    task += _cast_selection_rule(channel, text[:600])
    if hook.strip():
        task += ("\nThe story opens on this moment - make it beat 1: "
                 f'"{hook.strip()}"')
    revision = ""
    if feedback:
        revision = '\n\nThe user reviewed a previous version and asked for these changes: "' + feedback + '"'
        if previous:
            revision += ("\nPrevious beats:\n"
                         + "\n".join(f'{i}. [{b.get("setting", "")}] {b.get("action", "")}'
                                     for i, b in enumerate(previous, 1))
                         + "\nRevise them accordingly.")
    user = f"""{task}{revision}

The story is told in {n} BEATS. Each beat becomes ONE {secs}-second animated shot, in order, and
each shot is rendered starting from the LAST FRAME of the one before it - so consecutive beats share
a place, or the characters walk, ride, climb or open a door into the next one. Never a cut.

Return JSON with keys:
- "title": short title.
- "premise": 2-3 sentences telling the whole story in order (beginning, middle, end).
- "beats": exactly {n} objects, in story order, each with:
    - "setting": where this beat happens, 3-8 words ("on the sunny platform", "inside the carriage").
    - "action": ONE sentence - what HAPPENS. A physical, visible action with a clear start and end
      ("Ellie steps up onto the train"), never a feeling, a summary or a topic.
    - "characters": the names of the cast characters present in this beat (may be empty).
- "style": {_style_rule(art_style, channel)}

{BEAT_RULES}"""
    out = _chat([{"role": "system", "content": _system(channel, brief=True)},
                 {"role": "user", "content": user}])
    plan = {"title": str(out.get("title") or "").strip(),
            "premise": str(out.get("premise") or "").strip(),
            "style": str(out.get("style") or "").strip(),
            "beats": _as_beats(out.get("beats"), n)}
    plan["cast_chosen"] = _chosen(channel, out.pop("characters", None))
    if len(plan["beats"]) < config.STORY_MIN_BEATS:
        raise RuntimeError(f"vLLM returned only {len(plan['beats'])} story beats "
                           f"(at least {config.STORY_MIN_BEATS} are needed)")
    return plan


def _as_beats(value, n: int) -> list:
    """The reply's "beats" normalised to at most n well-formed beats.

    Short replies are kept as they are (the clip grid follows the beats, so a 9-beat story is simply
    a slightly shorter video); long ones are cut, because every extra beat is another GPU render."""
    out = []
    for item in (value or [])[:n]:
        if isinstance(item, str):
            item = {"action": item}
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or item.get("beat") or "").strip()
        if not action:
            continue
        chars = item.get("characters")
        if isinstance(chars, str):
            chars = re.split(r"[,;/]| and ", chars)
        out.append({"setting": str(item.get("setting") or "").strip(),
                    "action": action,
                    "characters": [str(c).strip() for c in (chars or []) if str(c).strip()]})
    return out


NO_DESCRIPTION_RULE = """THE MOST IMPORTANT RULE: these words are SPOKEN ALOUD over a picture the viewer can already see.
They must NEVER describe that picture. Never say what anyone wears, what colour or size anything is,
what species or age someone is, where the camera is, or what the scene looks like.
  WRONG: "Ellie the elephant, wearing a yellow dress with a long nose and blue trousers, is riding a blue train."
  RIGHT: "Ellie waited for the train to arrive."
  RIGHT: "All aboard!"
  RIGHT: "Ellie boarded the train and looked around."
Write it the way a picture book is read out loud: short sentences that move the story forward, plus
the characters' own words."""


def _beat_list(beats: list) -> str:
    return "\n".join(
        f'{i}. [{b.get("setting", "")}] {b.get("action", "")}'.replace("[] ", "")
        for i, b in enumerate(beats, 1))


def write_story_lines(beats: list, speakers: list, channel: dict | None = None,
                      used: list | None = None, max_words: int = 10) -> tuple:
    """The VOICEOVER of a story video, derived from the beats: one spoken line per beat.

    Returns (lines, issues), where lines is EXACTLY len(beats) objects {"speaker", "text"}. The
    count and the speaker roster are guaranteed here, deterministically, not by the model: a reply
    with the wrong count is padded or cut, and a speaker outside `speakers` becomes the narrator
    (their quoted words survive in the text, which is how a one-off conductor still gets to talk).

    The system prompt is built with spoken=True, so the cast sheet carries no looks at all: a writer
    that was never shown the look cannot recite it."""
    n = len(beats)
    roster = list(speakers) + [config.STORY_NARRATOR]
    user = f"""Below are the {n} beats of a story, in order. Write the words the VIEWER HEARS over each beat.

BEATS:
{_beat_list(beats)}

Return JSON with one key:
- "lines": exactly {n} objects, one per beat, in the same order, each with
    "speaker": one of {', '.join(roster)}, and
    "text": what is heard during that beat - one sentence, at most {max_words} words.

{NO_DESCRIPTION_RULE}

MORE RULES:
- A line is either (a) a character speaking, where "speaker" is that character and "text" is ONLY
  the words they say, with no "said Ellie" tag, or (b) {config.STORY_NARRATOR} telling the story,
  who MAY quote anyone who is not in the cast: "All aboard!" called the conductor.
- Only these may be a speaker: {', '.join(roster)}. Everyone else speaks through {config.STORY_NARRATOR}.
- Read end to end, the {n} lines are ONE continuous story: each follows from the last, nothing is
  repeated, nothing is skipped, and the last line ends it.
- Plain spoken English for a text-to-speech voice: no stage directions, no headings, no emojis, no
  asterisks, no "(excitedly)", no written-out sound effects.
- A purely visual beat may have an empty "text". Use this sparingly.
- {_narration_rule(channel)}"""
    out = _chat([{"role": "system", "content": _system(channel, used=used, spoken=True)},
                 {"role": "user", "content": user}])
    return _as_lines(out.get("lines"), n, speakers)


def _as_lines(value, n: int, speakers: list) -> tuple:
    """The reply's "lines" normalised to exactly n entries on a closed speaker roster.

    Returns (lines, issues). A short reply is padded with silent beats rather than retried: the
    beats still carry the story visually, and the user reads every line before approving anyway."""
    issues, out = [], []
    roster = list(speakers) + [config.STORY_NARRATOR]
    for item in (value or [])[:n]:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            item = {}
        text = re.sub(r"\s+", " ", str(item.get("text") or item.get("line") or "")).strip()
        want = str(item.get("speaker") or "").strip()
        speaker = next((s for s in roster if s.lower() == want.lower()), "")
        if want and not speaker:
            issues.append(f'"{want}" is not in this video - their line is told by '
                          f"{config.STORY_NARRATOR}")
        out.append({"speaker": speaker or config.STORY_NARRATOR, "text": text})
    if len(out) < n:
        issues.append(f"the writer returned {len(out)} lines for {n} beats - "
                      f"the last {n - len(out)} play without speech")
        out += [{"speaker": config.STORY_NARRATOR, "text": ""} for _ in range(n - len(out))]
    return out, issues


def rewrite_story_line(line: dict, beat: dict, reason: str, max_words: int,
                       channel: dict | None = None, used: list | None = None,
                       speakers: list | None = None) -> dict:
    """One line, written again - the only retry a story line ever gets.

    reason "describes": a deterministic check (services/story.describes_look) found the line saying
    out loud what the viewer can already see. reason "long": spoken, it ran past the longest clip a
    single line may buy. Either way only this line is rewritten, so the rest of the story - which
    the user may already have read - is untouched."""
    speakers = list(speakers or [])
    why = ("This line describes the picture instead of telling the story. The viewer can already SEE "
           "all of that." if reason == "describes" else
           "This line takes too long to say out loud. It must be shorter, without losing what "
           "happens in it.")
    user = f"""One line of a story voiceover has to be written again.

The beat it is heard over (what the viewer SEES): [{beat.get("setting", "")}] {beat.get("action", "")}
The line as written, spoken by {line.get("speaker") or config.STORY_NARRATOR}:
"{line.get("text", "")}"

WHY IT MUST CHANGE: {why}

Write the line again: at most {max_words} words, one sentence, same speaker, same moment of the story.

{NO_DESCRIPTION_RULE}

Return JSON: {{"text": "the new line"}}"""
    out = _chat([{"role": "system", "content": _system(channel, used=used, spoken=True)},
                 {"role": "user", "content": user}], max_tokens=1000)
    text = re.sub(r"\s+", " ", str(out.get("text") or "")).strip()
    return {"speaker": line.get("speaker") or config.STORY_NARRATOR, "text": text or line.get("text", "")}


def write_scenes(narration: str, style: str, n: int, on_progress=None,
                 kind: str = "voiceover narration", segments: list | None = None,
                 on_review_issue=None, channel: dict | None = None,
                 opening: dict | None = None, used: list | None = None,
                 timelines: list | None = None, clip_secs: list | None = None,
                 aligned: bool = True, beats: list | None = None, model: str = "") -> list:
    """Step 2: exactly n scene prompts, generated ONE PER CALL so scene i cannot drift
    away from narration segment i (a single big call loses sync on long lists).

    segments: optional pre-computed per-clip text (e.g. whisper-timed lyrics); empty
    entries mean nothing is heard during that clip (instrumental/silence).
    timelines: optional per-clip lyric timelines (lyrics.format_timeline), times relative to
    the clip start, so the storyboard beats can follow the singing.
    clip_secs: the real length of each clip (config.clip_durations); default: CLIP_SECONDS as
    `model` quantises it, for every clip.
    aligned: False when the segment text is raw speech recognition (may contain mis-hearings).
    opening: the cast character whose portrait is clip 1's first frame, if any.
    used: the names of this video's cast (plan["cast_used"]); None = legacy plan, all cast.
    beats: story mode - the {setting, action, characters} this clip must SHOW. The beat is the
    subject of the shot; the segment and its timeline only say when the words land on it."""
    if segments is None:
        segments = split_narration(narration, n)
    segments = (list(segments) + [""] * n)[:n]
    timelines = (list(timelines or []) + [""] * n)[:n]
    beats = (list(beats or []) + [None] * n)[:n]
    clip_secs = (list(clip_secs or []) + [config.actual_clip_seconds(model=model)] * n)[:n]
    cast_rule = ""
    roster = cast.active(channel)
    if roster:
        in_video = cast.names(roster) if used is None else list(used)
        if in_video:
            cast_rule = (f"\nCAST: only {', '.join(in_video)} appear in this video. Whenever one of them is on "
                         "screen, name them and describe them exactly as in the cast sheet (species/age, "
                         "colours, clothes, marks). Do not add any other cast member.")
        else:
            cast_rule = ("\nCAST: none of the channel's recurring characters appear in this video - do not "
                         "add them.")
    elif not cast.enabled(channel):
        cast_rule = NO_CHARACTER_SCENE_RULE
    first = ("This is the FIRST clip of the video: establish the world." if cast.enabled(channel)
             else "This is the FIRST clip of the video: establish the subject and the setting.")
    if opening:
        first = (f"This is the FIRST clip of the video. Its first frame is a fixed reference image of "
                 f"{opening['name']} alone, standing centered and facing the viewer in a simple soft "
                 f"setting ({opening.get('look', '').strip()}). Start EXACTLY from that image: {opening['name']} "
                 "comes alive, and the story begins around them - do not open with a different shot.")
    scenes = []
    for i, seg in enumerate(segments):
        if on_progress:
            on_progress(f"scene {i + 1}/{n}")
        prev = (f'The previous clip\'s scene was:\n"{scenes[-1]}"\n'
                "MANDATORY: this clip's [0s] beat continues EXACTLY from that scene's ending frame - same "
                "camera angle, same characters in the same positions, same setting and lighting. Do not cut "
                "to a new shot or angle at the start; evolve the action from that frozen moment.") if scenes \
            else first
        nxt = (f'For flow only, the next segment will be: "{segments[i + 1]}"') if i + 1 < n \
            else "This is the FINAL clip: give it a conclusive ending."
        secs = _secs(clip_secs[i])
        if beats[i]:
            b = beats[i]
            who = ", ".join(b.get("characters") or []) or "no cast character"
            spoken = (f"""While this clip plays, the viewer HEARS exactly this, and nothing else:
"{seg}"
TIMELINE - when the words are heard, measured from the START of this clip:
{timelines[i]}""" if seg.strip() else
                      "No one speaks during this clip. Carry the beat visually: continue the previous "
                      "shot's motion and let the action play out in silence.")
            heard = f"""This clip is BEAT {i + 1} of the story:
  Where: {b.get("setting", "") or "as the previous clip"}
  What happens: {b.get("action", "")}
  Who is there: {who}
Build the whole shot around that action: it must visibly start, happen and finish inside the {secs} seconds.

{spoken}

The spoken words are the SOUNDTRACK, not the subject. Never render them as on-screen text or
subtitles, and never add a narrator, storyteller, book, microphone or stage to illustrate the
speaking - a cast character already on screen MAY be shown talking while their own line plays. Your
storyboard timestamps follow the TIMELINE: the key moment of the action lands while its words are
heard, and "(no speech)" spans carry motion, reaction or atmosphere and introduce nothing that
belongs to another beat."""
        elif seg.strip() and timelines[i]:
            heard = f"""While this clip plays, the {kind} is EXACTLY this (the lyric text, timed from the recording):
"{seg}"
TIMELINE - what is sung, with times measured from the START of this clip:
{timelines[i]}

Write ONE scene prompt for this clip that visualizes exactly what is heard during it.
EVERY object, creature, letter or concept mentioned above MUST appear in the storyboard - give each
its own moment within the {secs} seconds. Your storyboard timestamps MUST follow the TIMELINE: the
visual beat for a sung line begins when that line begins and lasts while it is sung; "(instrumental)"
spans carry motion, reaction or atmosphere and introduce nothing new. Do NOT add a singer, band,
microphone or stage to illustrate the singing; if a cast character already on screen fits a line,
they MAY be shown singing or mouthing it, but never invent a performer for that purpose. Never
render the lyrics as on-screen text. Do not show things from other segments except as a
continuation of the previous shot."""
        elif seg.strip():
            heard = f"""While this clip plays, the {kind} is EXACTLY this:
"{seg}"

Write ONE scene prompt for this clip that visualizes exactly what is heard during it.
EVERY object, creature, letter or concept mentioned above MUST appear in the storyboard —
give each its own moment within the {secs} seconds. Do not show things from other segments
except as a continuation of the previous shot."""
        else:
            heard = f"""During this clip the {kind} is INSTRUMENTAL - no words are heard.
Write ONE scene prompt that carries the story visually: continue the previous shot's motion,
build atmosphere, or transition toward the next segment. Introduce nothing that belongs to
other segments' content."""
        user = f"""Global visual style: {style}

We are storyboarding clip {i + 1} of {n} for a video. This clip is {secs} seconds long.
The soundtrack is a {kind}.
{prev}
{nxt}
{heard}
{SCENE_RULES.format(secs=secs)}{cast_rule}
Return JSON: {{"scene": "the scene prompt string"}}"""
        out = _chat([{"role": "system", "content": _system(channel, used=used)},
                     {"role": "user", "content": user}])
        scene = out.get("scene") or next(iter(_as_strings(out.get("scenes", []))), "")
        if isinstance(scene, dict):
            scene = scene.get("prompt", json.dumps(scene))
        if not str(scene).strip():
            raise RuntimeError(f"vLLM returned no scene for segment {i + 1}")
        scene = str(scene).strip()
        if config.SCRIPT_REVIEW:
            if on_progress:
                on_progress(f"reviewing scene {i + 1}/{n}")
            scene, issues = review_scene(scene, seg, style, i + 1, n, kind=kind,
                                         prev_scene=scenes[-1] if scenes else "",
                                         channel=channel, used=used,
                                         timeline=timelines[i], aligned=aligned,
                                         beat=(beats[i] or {}).get("action", "") if beats[i] else "")
            if issues and on_review_issue:
                on_review_issue(i + 1, issues)
        scenes.append(scene)
    return scenes


SAFETY_CHECKS = {
    "kids": "Content is gentle and safe for young children.",
    "family": "Content is suitable for a family audience (no gore, explicit content or profanity).",
    "general": "Content follows YouTube community and advertiser guidelines.",
}


def review_scene(scene: str, segment: str, style: str, i: int, n: int,
                 kind: str = "voiceover narration", prev_scene: str = "",
                 channel: dict | None = None, used: list | None = None,
                 timeline: str = "", aligned: bool = True, beat: str = "") -> tuple:
    """Second-pass review of one scene prompt against the audio it must match.

    Returns (corrected_scene, issues). Catches inconsistencies like a letter or object
    in the storyboard not matching what the audio actually says at that moment, and - when a
    lyric timeline is given - storyboard beats that do not fall where their line is sung.

    beat: story mode - the action this clip must show. The scene is then checked against the BEAT
    (the spoken line is only the soundtrack), and the sung-lyrics wording is dropped."""
    if beat.strip():
        audio_ctx = (f"This clip must show one beat of a story:\n\"{beat.strip()}\"")
        if segment.strip():
            audio_ctx += f'\nWhile it plays the viewer hears exactly this, and nothing else:\n"{segment}"'
            if timeline:
                audio_ctx += f"\nTIMELINE (when the words are heard, times from the START of this clip): {timeline}"
        else:
            audio_ctx += "\nNobody speaks during this clip."
        rule = ("The storyboard must show that beat happening, from start to finish, inside this clip. "
                "Everything it shows must be consistent with the beat and with what is heard; the "
                "spoken words are never drawn as on-screen text or subtitles, and no narrator, "
                "storyteller, book, microphone or stage is added to illustrate the speaking.")
    elif segment.strip():
        audio_ctx = f'While this clip plays, the {kind} is EXACTLY this:\n"{segment}"'
        if timeline:
            audio_ctx += f"\nTIMELINE (what is sung, times from the START of this clip): {timeline}"
        rule = ("Every letter, word, object, creature, name and count that the storyboard displays or "
                "features MUST match what the audio says during this clip - e.g. if the audio says "
                "\"F is fairy\", the scene must show the letter F with the fairy, never a different letter.")
        if not aligned:
            rule += (" IMPORTANT: the audio text comes from automatic speech transcription and may contain "
                     "small mis-hearings - if it pairs a letter with a word that clearly starts with a DIFFERENT "
                     "letter (e.g. \"A is for fairy\"), trust the word: fairy belongs to the letter F, so show F.")
    else:
        audio_ctx = f"During this clip the {kind} is INSTRUMENTAL - no words are heard."
        rule = "The scene must not introduce letters/objects that belong to other clips' audio."
    timing = ""
    if beat.strip():
        if timeline and segment.strip():
            timing = ("5. TIMING: the key moment of the action falls within about 1 second of the words that "
                      "go with it in the TIMELINE; beats during \"(no speech)\" spans carry motion, reaction "
                      "or atmosphere and introduce nothing that belongs to another part of the story. The "
                      "timeline is timing metadata only: never add the spoken words as on-screen text.")
    elif timeline and segment.strip():
        timing = ("5. TIMING: every storyboard beat that visualises a sung line falls within about 1 second of "
                  "that line's window in the TIMELINE; beats during \"(instrumental)\" spans introduce nothing "
                  "that belongs to a later line. The timeline is timing metadata only: never add the lyrics as "
                  "on-screen text or subtitles, and never invent a singer, band, microphone or stage.")
    continuity = ""
    if prev_scene.strip():
        continuity = f"""6. SEAMLESS CONTINUITY: this clip is rendered starting from the LAST frame of the previous clip.
The previous clip's scene was:
\"\"\"{prev_scene}\"\"\"
This scene's [0s] beat must directly continue that scene's ending (same camera angle, same characters in
the same positions, same setting); it must NOT open with a cut, a new shot, a different angle or location.
Reframing may only happen through continuous camera motion within the clip."""
    safety = SAFETY_CHECKS[_channel(channel)["audience"]]
    if cast.active(channel):
        safety += (" Cast characters keep their canonical name and look from the cast sheet; no unnamed "
                   "substitute animals or kids where a cast character fits")
        if used is not None:
            safety += (f"; no cast members other than {', '.join(used)}" if used
                       else "; none of the channel's cast members appear in this video")
        safety += "."
    elif not cast.enabled(channel):
        safety += (" This channel has no characters: no named person, no protagonist and no story arc - "
                   "anyone on screen is generic and unnamed (role, rough age and clothing only), and "
                   "the subject of the shot is the topic itself.")
    user = f"""You are reviewing a scene prompt for clip {i} of {n} of a video before it is rendered.
Global visual style: {style}

{audio_ctx}

Scene prompt to review:
\"\"\"{scene}\"\"\"

Check strictly:
1. {rule}
2. Internal consistency (no contradictions within the storyboard).
3. {safety}
4. The required format is kept (overview, timestamped storyboard, camera, audio, the closing no-text line).
{timing}
{continuity}

If everything is correct, return the scene UNCHANGED. If there are issues, FIX them with the smallest
possible edits, keeping everything else identical.
Return JSON: {{"issues": ["short description of each issue found, empty list if none"], "scene": "the final scene prompt"}}"""
    out = _chat([{"role": "system", "content": _system(channel, used=used)}, {"role": "user", "content": user}],
                max_tokens=4000)
    fixed = str(out.get("scene") or "").strip()
    issues = [str(x) for x in (out.get("issues") or []) if str(x).strip()]
    if not fixed:
        return scene, []
    return fixed, issues


def review_script(narration: str, context: str = "", channel: dict | None = None,
                  verbatim: bool = False) -> tuple:
    """One review pass over a drafted script. Returns (corrected_narration, issues).

    verbatim: the user pasted these exact words - their wording wins, so the appearance check
    (which rewrites sentences) is left out."""
    ctx = f"\nContext: {context}" if context else ""
    safety = {"kids": "anything not gentle and kid-safe",
              "family": "anything unsuitable for a family audience",
              "general": "anything violating YouTube community or advertiser guidelines"}[
        _channel(channel)["audience"]]
    cast_checks = ", stray cast names (only the characters that belong to this story appear)" \
        if cast.active(channel) else ""
    if cast_checks and not verbatim:
        cast_checks += (", character looks in the spoken words (the script names a character and never "
                        "describes them: strip any fur, colour, clothing, accessory or marking detail "
                        'hung on a name - "Piko, a small round panda in a yellow T-shirt, is worried" '
                        'becomes "Piko is worried" - and change nothing else)')
    elif not cast.enabled(channel) and not verbatim:
        cast_checks = (", invented characters (this channel has none: no named person, no protagonist, "
                       "no 'meet Alex' - rewrite any such passage to address the viewer as 'you' or to "
                       "name an unnamed role, keeping the same information), and explainer shape (hook, "
                       "the viewer's problem, the concrete steps in order, then the payoff - all in the "
                       "second person)")
    user = f"""You are reviewing the final voiceover script of a video before production.{ctx}

Script:
\"\"\"{narration}\"\"\"

Check strictly: factual slips, internal inconsistencies (e.g. a letter or item paired with the wrong
word), ordering mistakes, {safety}, awkward phrasing that would sound wrong
read aloud{cast_checks}.
If everything is correct, return the script UNCHANGED; otherwise fix it with minimal edits.
Return JSON: {{"issues": ["each issue found, empty list if none"], "narration": "the final script"}}"""
    out = _chat([{"role": "system", "content": _system(channel, spoken=True)}, {"role": "user", "content": user}],
                max_tokens=4000)
    fixed = str(out.get("narration") or "").strip()
    issues = [str(x) for x in (out.get("issues") or []) if str(x).strip()]
    if not fixed:
        return narration, []
    return fixed, issues


def _storyboard_subjects(channel: dict | None) -> str:
    """What a story-board card's still image may show."""
    return ("an object, device, place, abstract scene or one of the channel's cast characters"
            if cast.enabled(channel) else
            "an object, device, place, screen, diagram or abstract scene, never a named character")


def write_storyboard(title: str, narration: str, segments: list, n: int,
                     channel: dict | None = None, max_images: int | None = None,
                     clip_seconds: float | None = None, used: list | None = None) -> list:
    """Plan the on-screen story board cards: short facts/takeaways shown as styled text over
    the video, each tied to a range of clips, some with a related still image.

    Returns the raw card list (validated by storyboard.normalize_cards)."""
    max_images = config.STORYBOARD_MAX_IMAGES if max_images is None else max_images
    secs = clip_seconds or config.CLIP_SECONDS
    segs = (list(segments) + [""] * n)[:n]
    timeline = "\n".join(f'clip {i + 1} ({secs * i:.0f}-{secs * (i + 1):.0f}s): "{seg}"'
                         for i, seg in enumerate(segs))
    img_rule = (f'- "image_prompt": for AT MOST {max_images} cards (the ones where a picture adds the '
                f'most), a prompt for ONE photorealistic still image that illustrates the card\'s fact '
                f'({_storyboard_subjects(channel)} - never '
                f'a real person\'s face). The image generator '
                f'CANNOT render writing: the prompt must not ask for any words, labels, captions, UI text, '
                f'code on screens or lettering of any kind. Empty string "" for the other cards.'
                if max_images > 0 else '- "image_prompt": always an empty string "".')
    user = f"""We produced a short video titled "{title}" with this voiceover:
\"\"\"{narration}\"\"\"

The video is {n} clips of {secs:g} seconds; what is heard during each clip:
{timeline}

Design the on-screen STORY BOARD for it: styled text cards shown over the video (like the
fact cards and lower-thirds of a tech explainer) that give viewers concrete, accurate
facts, definitions, numbers and takeaways matching what is being said at that moment.
Rules:
- Between {max(2, n // 3)} and {max(3, (2 * n) // 3)} cards; each covers ONE clip or TWO consecutive clips
  (5-10 seconds on screen); cards never overlap; keep some clips card-free for breathing room.
- The FIRST clip is the intro: its card (if any) names the topic. The LAST clip usually gets a
  short takeaway or call-to-action card.
- Text is READ ON SCREEN, so keep it short and punchy: "headline" max 8 words, "body" one
  sentence of max 18 words. Facts must be true and match the narration; no hype, no emojis.
Return JSON: {{"cards": [{{"start_clip": 1, "end_clip": 1, "kicker": "...", "headline": "...", "body": "...", "image_prompt": "..."}}, ...]}} where:
- "kicker": a 1-3 word uppercase label such as KEY FACT, DID YOU KNOW, HOW IT WORKS, BY THE NUMBERS, PRO TIP, TAKEAWAY.
- "headline": the card's main line.
- "body": the supporting sentence (may be "" when the headline says it all).
{img_rule}
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, used=used)}, {"role": "user", "content": user}])
    cards = out.get("cards")
    if not isinstance(cards, list) or not cards:
        raise RuntimeError("vLLM returned no story board cards")
    return cards


PLAN_TITLE_KEY = '- "title": the video title a viewer would click, max 12 words.\n'

# The planner's titles are the first draft of what write_seo later polishes, so a kids week that
# starts with clever titles spends the SEO call fixing them instead of sharpening them.
PLAN_TITLE_KEY_KIDS = ('- "title": the video title a parent would click, max 10 words, OPENING with the '
                       'character name and the plain subject a parent would type ("Train", "Clean Up '
                       'Song", "Learn Colors", "ABCs", "Bedtime", "Counting"). No clever framing '
                       'before the subject, no brackets, no number first.\n')

PLAN_ROW_KEYS = ('- "actors": the cast characters of this video, by name, as a list (usually 1-3; '
                 'empty when the story needs none).\n'
                 '{title}'
                 '- "description": 2-4 sentences telling the whole story from beginning to end - what '
                 'happens, to whom, and how it ends. This becomes the brief the script is written from, '
                 'so name the characters and never describe how they look.\n'
                 '- "thumbnail_idea": one sentence describing the thumbnail picture (who is in it, what '
                 'they are doing, the feeling).\n'
                 '- "hook": the spoken opening line of the video, max 15 words, in quotation marks in the '
                 'story\'s own voice.')


PLAN_ROW_KEYS_NO_CHARACTERS = (
    '{title}'
    '- "description": 2-4 sentences describing the whole video from beginning to end - what the '
    'viewer learns, the concrete steps, settings or demo it walks through, in order, and what they '
    'can do at the end. This becomes the brief the script is written from: it is an informative '
    'explainer addressed to "you", so never invent a person, a name, a protagonist or a story - '
    'never "Alex, a project manager who is overwhelmed by 50 unread emails".\n'
    '- "thumbnail_idea": one sentence describing the thumbnail picture - the object, screen or moment '
    'it shows and the feeling it gives (an unnamed person may be in frame, never the point).\n'
    '- "hook": the spoken opening line of the video, max 15 words, in quotation marks, spoken '
    'straight to the viewer.')


def _plan_row_keys(channel: dict | None) -> str:
    shape = PLAN_ROW_KEYS if cast.enabled(channel) else PLAN_ROW_KEYS_NO_CHARACTERS
    kids = _channel(channel)["audience"] == "kids"
    return shape.format(title=PLAN_TITLE_KEY_KIDS if kids else PLAN_TITLE_KEY)


def _plan_shape(channel: dict | None, day_value: str) -> str:
    """The JSON row shape asked for. "actors" is dropped entirely for a channel with no characters,
    so the writer is never shown a slot to fill."""
    actors = '"actors": ["..."], ' if cast.enabled(channel) else ""
    return ('{"day": ' + day_value + ', ' + actors + '"title": "...", "description": "...", '
            '"thumbnail_idea": "...", "hook": "..."}')


DAYS_PER_WEEK = 7          # a full week; fewer videos than this are spread across it

STORY_WEEK_RULE = (
    "These are STORY videos. Each one is told as a short series of scenes with NO CUTS between them: "
    "the camera never jumps, so the characters walk, ride, climb or open a door from one place into "
    "the next. Plan a story that happens in one or two connected places and resolves by the end - "
    "not one that moves across the world.")


def _plan_rules(channel: dict | None, days: int, week_start: str, notes: str, avoid,
                video_mode: str = "song") -> str:
    """The shared part of the weekly-plan prompts: what a good week looks like.

    video_mode: "story" adds the one constraint a story brief has and a song brief does not -
    a story is rendered as clips that each continue from the previous one's last frame, so a
    week of ideas that hop between unconnected places fights the continuity contract."""
    avoid = [str(a).strip() for a in (avoid or []) if str(a).strip()]
    cadence = "one per day" if days >= DAYS_PER_WEEK else "spread across the week"
    parts = [f"Plan {days} videos for this channel, {cadence}"
             + (f", starting {week_start}." if week_start else ".")]
    if days < DAYS_PER_WEEK:
        # A short week is a deliberate choice, not a shortage of ideas: uploading less and better
        # is what the channel's own numbers asked for, so the brief has to raise the bar per video
        # rather than hand back three of the seven it would have written.
        parts.append(f"This channel publishes {days} videos a week, not one a day - so these are the "
                     "week's BEST ideas, not a thin spread. Each one has to be worth a viewer's full "
                     "attention: a complete little story or lesson with a real beginning and end, not "
                     "a fragment that would have been one day of seven.")
    if cast.enabled(channel):
        parts.append("Every video stands on its own (a viewer may watch only one) but the week as a "
                     "whole has variety: different settings, different problems to solve, different "
                     "characters in the lead. Never plan the same story twice.")
        if cast.active(channel):
            parts.append("Cast the videos from the channel roster above - usually 1-3 characters each, "
                         "and not the same lead every day. Use their names exactly as written.")
            if cast.leads(channel):
                parts.append("Most of the week belongs to the roster's LEAD characters: give each video "
                             "a lead in it, and a character marked resting none at all unless this "
                             "request names them.")
            if video_mode == "story":
                parts.append(STORY_WEEK_RULE)
    else:
        parts.append("Every video stands on its own (a viewer may watch only one) but the week as a "
                     "whole has variety: a different concrete problem, a different tool or workflow, a "
                     "different angle on the channel's subject each day. Never plan the same video "
                     "twice.")
        parts.append("This channel has NO characters and tells NO stories: every video is an "
                     "informative explainer spoken to the viewer in the second person. Do not invent "
                     "people, names, protagonists or narratives - plan the skills, problems, tools, "
                     "workflows and step-by-step demos a viewer can follow themselves.")
    if notes.strip():
        parts.append(f'The user asks for this week: "{notes.strip()}"')
    if avoid:
        parts.append("These titles were already planned for this channel - plan something different:\n- "
                     + "\n- ".join(avoid[:40]))
    return "\n\n".join(parts)


def write_week_plan(channel: dict | None = None, days: int = 7, week_start: str = "",
                    notes: str = "", avoid=(), video_mode: str = "song") -> list:
    """A week of video ideas: one row per day (actors, title, description, thumbnail idea, hook).

    Returns the raw row list (normalized by services.planner.normalize_days). The rows are written
    with the spoken cast sheet: a description becomes the brief the narration is written from, so it
    must not carry the characters' looks."""
    user = f"""{_plan_rules(channel, days, week_start, notes, avoid, video_mode)}

Return JSON: {{"plan": [{_plan_shape(channel, '1')}, ...]}} with exactly {days} entries, "day" running 1 to {days}, where:
{_plan_row_keys(channel)}
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, spoken=True)},
                 {"role": "user", "content": user}])
    rows = out.get("plan") or out.get("days") or out.get("videos")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("vLLM returned no weekly plan")
    return rows


def write_plan_day(channel: dict | None = None, day: int = 1, others: list | None = None,
                   feedback: str = "", days: int = 7, week_start: str = "", notes: str = "",
                   avoid=(), video_mode: str = "song") -> dict:
    """One replacement row for day `day`, written so it stays different from the other days."""
    rest = "\n".join(f'day {r.get("day")}: {r.get("title", "")} - {r.get("description", "")}'
                     for r in (others or []) if r.get("title") or r.get("description"))
    context = f"\n\nThe other days of this week are:\n{rest}\n\nThe new idea must be different from all of them." \
        if rest else ""
    wish = f'\n\nThe user asks for this day: "{feedback.strip()}"' if feedback.strip() else ""
    user = f"""{_plan_rules(channel, days, week_start, notes, avoid, video_mode)}{context}{wish}

Rewrite DAY {day} only, as one fresh idea.
Return JSON: {{"day": {_plan_shape(channel, str(day))}}} where:
{_plan_row_keys(channel)}
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, spoken=True)},
                 {"role": "user", "content": user}])
    row = out.get("day") or out.get("plan") or out
    if isinstance(row, list):
        row = next(iter(row), {})
    if not isinstance(row, dict) or not str(row.get("title") or "").strip():
        raise RuntimeError(f"vLLM returned no plan for day {day}")
    return row


_SEO_TITLE_RULE = ('- "seo_title": one compelling, keyword-rich title, max 70 characters, no clickbait '
                   'lies, no quotes, no emojis at the start (one emoji at the end is allowed if it fits '
                   'the topic).')

# A kids title is not read, it is scanned - by a parent typing two or three words into the search
# box and by a sidebar that truncates at about 60 characters on a phone. So the words that get the
# video found have to come FIRST, before any framing, and the clever part has to go or go last.
_SEO_TITLE_RULE_KIDS = (
    '- "seo_title": max 60 characters, and the first 40 must already carry the words a parent would '
    'type - the character\'s name and the plain subject ("Train", "Clean Up Song", "Learn Colors", '
    '"ABCs", "Bedtime", "Counting"). Start with those words: never open with the channel name, a '
    'number, a bracket or a clause before the subject. Plain, concrete, no clickbait, no quotes, no '
    'emoji at the start. A short "| Nursery Rhymes & Kids Songs"-style tail after the subject is '
    'fine if it fits the 60 characters.')


def _title_rule(channel: dict | None) -> str:
    """The seo_title clause: audience-specific, because what earns a click differs by who clicks."""
    return _SEO_TITLE_RULE_KIDS if _channel(channel)["audience"] == "kids" else _SEO_TITLE_RULE


def write_seo(title: str, narration: str, transcript: str, channel: dict | None = None,
              used: list | None = None) -> dict:
    """YouTube-ready SEO title and description for the generated video."""
    name = _channel(channel)["name"]
    user = f"""We produced a short AI-generated video titled "{title}".
What the video says (its narration/story):
\"\"\"{narration}\"\"\"

Original source transcript excerpt (for topic keywords):
\"\"\"{transcript[:4000]}\"\"\"

Write YouTube upload metadata optimized for SEO and reach. Return JSON with keys:
{_title_rule(channel)}
- "seo_keywords": 5-8 search phrases people would type to find this video (the main topic first, then related and long-tail phrases).
- "seo_description": a description of 2-3 short paragraphs (120-200 words total). The first sentence hooks and contains the main keyword. Work every keyword from "seo_keywords" into the text naturally, in the words a viewer would search for - repeat the main keyword 2-3 times, and never list keywords bare. End with a call-to-action inviting viewers to subscribe to "{name}", then a final line of 4-6 hashtags made from those keywords. Hashtags are all lowercase with no spaces, e.g. #spacefacts #kidslearning.
Never mention that the video is AI-generated unless the content is about AI. Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, used=used, rules=False)},
                 {"role": "user", "content": user}])
    # seo_keywords is asked for above and used to be thrown away here, so every upload went out
    # with no tags at all - the keywords now reach services/youtube.clean_tags.
    kws = out.get("seo_keywords")
    return {"seo_title": str(out.get("seo_title", "")).strip()[:100],
            "seo_keywords": [str(k).strip() for k in (kws or []) if str(k).strip()][:12],
            "seo_description": _lowercase_hashtags(str(out.get("seo_description", "")).strip())}


def write_compilation_seo(part_titles: list, length: str, channel: dict | None = None,
                          used: list | None = None) -> dict:
    """Title, description, keywords and thumbnail lines for a compilation of finished videos.

    A compilation is sold on two things a single video is not: how LONG it is (a parent putting it
    on wants to know it will not end in forty seconds) and WHAT IS IN IT (the list is the reason to
    click). So the length goes in the title and the parts go in the description. Nothing about the
    pictures is written here - the thumbnail is lifted from one of the parts, which already have
    one - only the lines drawn over it.

    `length` is services.compilation.fmt_length(total) - the model is given the length rather than
    asked to work it out, because an overstated one is the worst thing a compilation title can do."""
    from services import thumbtext   # lazy: thumbtext pulls storyboard, which imports this module

    ch = _channel(channel)
    max_words, max_chars = thumbtext.caps(ch)
    listing = "\n".join(f"- {t}" for t in (part_titles or []) if str(t or "").strip())
    user = f"""We joined {len(part_titles or [])} of this channel's finished videos into ONE
compilation that runs {length}. In order, the videos inside it are:
{listing}

Write YouTube upload metadata for the compilation. Return JSON with keys:
- "seo_title": max 70 characters, and it must OPEN with the length exactly as "{length}" followed by
  the subject the videos share - the recurring character, or what they teach. Then, if it fits, a
  short searchable tail after a "|". It is a compilation, so say so in the words a viewer searches
  for (compilation, non-stop, back to back, full episodes) rather than the word "playlist".
  Never promise more time than "{length}". No clickbait, no quotes, no emoji at the start.
- "seo_keywords": 5-8 search phrases someone would type to find a long compilation like this (the
  shared subject first, then the compilation and long-play phrasings, then the individual topics).
- "seo_description": 2-3 short paragraphs (120-200 words). The first sentence says what the
  compilation is, how long it runs and who it is for. Work the keywords in naturally. End with a
  call-to-action inviting viewers to subscribe to "{ch['name']}", then a final line of 4-6
  hashtags, all lowercase with no spaces. Do NOT list the videos or their timestamps - a chapter
  list is appended to this description automatically afterwards.
- "thumbnail_texts": 2-3 alternative overlay lines for the compilation's thumbnail, each at most
  {max_words} words and under {max_chars} characters, no full stops, no quotation marks. The length is
  the strongest thing one can say, so at least one of them should carry it (e.g. "{length}").
Never mention that the videos are AI-generated. Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, used=used, rules=False)},
                 {"role": "user", "content": user}])
    kws = out.get("seo_keywords")
    raw = out.get("thumbnail_texts") or out.get("thumbnail_text") or []
    if isinstance(raw, str):
        raw = [raw]
    texts, seen = [], set()
    for t in list(raw)[:4]:
        t = thumbtext.normalize(t, max_words, max_chars)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            texts.append(t)
    return {"seo_title": str(out.get("seo_title", "")).strip()[:100],
            "seo_keywords": [str(k).strip() for k in (kws or []) if str(k).strip()][:12],
            "seo_description": _lowercase_hashtags(str(out.get("seo_description", "")).strip()),
            "thumbnail_texts": texts}


_ALNUM = re.compile(r"[^a-z0-9]+")


def _quoted_from(narration: str, quote: str) -> bool:
    """Is the promise really in the script?

    Compared on letters and digits only, so smart quotes, hyphens and re-punctuation do not matter
    while a paraphrase still fails. The whole quote first, then its longest sentence - a quote may
    legitimately span a line break. Below ~25 characters nothing counts: a short fragment matches by
    accident, and a fabricated promise is the exact bug this guards against."""
    hay = _ALNUM.sub(" ", (narration or "").lower())
    for part in sorted([quote] + re.split(r"(?<=[.!?])\s+", quote), key=len, reverse=True):
        needle = _ALNUM.sub(" ", part.lower()).strip()
        if len(needle) >= 25 and needle in hay:
            return True
    return False


def write_resource(title: str, narration: str, source_text: str = "", channel: dict | None = None,
                   used: list | None = None, max_chars: int = 3000) -> dict:
    """The thing the video promised its viewer - or nothing at all.

    Decides whether the FINAL narration offers the viewer a concrete take-away ("the exact prompt
    structure is in the description", "grab the checklist") and, only then, writes that exact thing
    in full. The script is never touched and is never told to make such an offer - the promise is
    the writer's own; this only keeps the ones it made.

    An invented offer is the bug this exists to prevent, so the answer must quote the promise word
    for word out of the narration; a quote that is not in the narration is thrown away and nothing
    is produced. source_text is the README or transcript the video was written from: for repo and
    URL jobs the narration deliberately leaves out the detail a resource needs (see the repo arm of
    write_script_from_prompt), so without it the model would invent the template.

    Returns {} when the video promises nothing (the normal case), otherwise
    {"promise", "kind", "title", "markdown"}."""
    words = max(150, min(600, max_chars // 6))
    source = source_text[: config.MAX_TRANSCRIPT_CHARS] if source_text else ""
    extra = (f"""

For the facts, wording and detail - NOT for deciding whether a promise was made - here is the
source the video was written from:
\"\"\"{source}\"\"\"""" if source else "")
    user = f"""This is the FINAL voiceover script of a video titled "{title}". It is locked: the video
says exactly this, word for word, and nothing you write can change it.

Script:
\"\"\"{narration}\"\"\"{extra}

STEP 1 - DECIDE. Does this script PROMISE THE VIEWER A CONCRETE THING TO TAKE AWAY - a prompt, a
prompt structure, a template, a checklist, a cheat sheet, a script, a worksheet, a settings list, a
step-by-step recipe - whether it says the thing is "in the description", "linked below", "free to
download", or simply hands it over ("here is the exact prompt structure we used")? Read the script
literally: an offer is a promise of an ARTEFACT the viewer expects to receive.
These are NOT offers: teaching a technique, explaining how something works, naming tips inside the
narration with nothing to take away, "subscribe", "comment below", "watch the next one", "try it
yourself", or mentioning that such a thing exists somewhere in the world. If you are not certain the
script promises an artefact, it does not.

STEP 2 - PRODUCE, only when STEP 1 found a promise. Write EXACTLY the thing that was promised,
finished and usable as it stands:
- THE THING NAMED, never a summary of the video. "The exact prompt structure" means that prompt
  itself, written out in full and ready to paste. "The checklist" means boxes to tick. "The
  template" means the fill-in template with its placeholders. "The steps" means the numbered steps.
- Everything the script says about it must be true of it: every keyword, section, step, setting or
  number the video names appears in it, in the same order and in the video's own wording wherever
  the video spelled it out. Fill in the detail a viewer needs to use it; contradict the video nowhere.
- It works WITHOUT the video: no "as I mentioned", no timestamps, no "see above", no links.
- Plain Markdown, self-contained: one "# " title line, short "## " sections, bullet lists, numbered
  steps, "- [ ] " checkboxes, and ``` fenced blocks around anything meant to be copied verbatim
  (prompts, commands, code). No HTML, no images, no links, no tables - it is also read as plain text
  and printed as a PDF.
- Between 150 and {words} words and at most {max_chars} characters. Dense and usable: no
  introduction about the video, no filler, no sign-off.
- Name the tools, models and settings the video names, spelled the way the video spells them.
  Never mention that the video is AI-generated.

Return JSON with keys:
- "promise": the sentence or two from the script that make the offer, COPIED WORD FOR WORD out of
  the script above - never paraphrased, never reconstructed, never invented. "" if there is none.
- "kind": what the promised thing is, 1-3 lowercase words, as the script calls it ("prompt
  structure", "checklist", "template"). "" if there is none.
- "title": the resource's own title, max 10 words, naming the thing it is. "" if there is none.
- "resource": the complete Markdown of that thing. "" if there is none.

If the script promises the viewer nothing, return exactly
{{"promise": "", "kind": "", "title": "", "resource": ""}} and write no resource. That is the normal,
correct answer for most videos: an unasked-for resource is waste, and a resource for a promise the
video never made is a lie about what the viewer is getting."""
    out = _chat([{"role": "system", "content": _system(channel, used=used, rules=False)},
                 {"role": "user", "content": user}])
    promise = str(out.get("promise") or "").strip()
    body = str(out.get("resource") or "").strip()
    if not promise or not body or not _quoted_from(narration, promise):
        return {}     # nothing promised - or a promise the script never actually made
    kind = re.sub(r"\s+", " ", str(out.get("kind") or "")).strip().lower().strip(".")[:40]
    name = re.sub(r"\s+", " ", str(out.get("title") or "")).strip().strip(chr(34))[:120]
    return {"promise": promise[:500], "kind": kind or "resource",
            "title": name or f"The {kind or 'resource'} from this video", "markdown": body}


def write_repo_guide(title: str, narration: str, source_text: str, source_url: str = "",
                     channel: dict | None = None, used: list | None = None,
                     max_chars: int = 7000) -> dict:
    """The hands-on guide that ships with every repo / Hugging Face video.

    Unlike write_resource this asks no question and keeps no promise. A video built from a README
    always hands its viewer the setup guide for that project: the README it was written from
    already holds the install and quickstart material, and the video is the walkthrough. Nobody was
    told to look in the description for it, so the guide is a download only.

    The narration is supplied for wording and ordering - the guide must never contradict what the
    viewer just heard - but every command, flag and version comes from source_text. The model's own
    memory of a project is exactly the thing that invents a pip package that does not exist, so the
    prompt forbids it, the same way the repo arm of write_script_from_prompt does.

    Returns {"kind", "title", "markdown"}, or {} when nothing usable came back - a missing guide is
    never worth failing a video for."""
    words = max(300, min(1000, max_chars // 7))
    source = (source_text or "")[: config.MAX_TRANSCRIPT_CHARS]
    if not source.strip():
        return {}
    page = f'\nThe project page is {source_url} .' if source_url else ""
    user = f"""Below is the README and the project facts of the project a video titled "{title}" was
made from, and the video's FINAL voiceover script. Write the hands-on guide that goes out with that
video - the thing a viewer keeps open in a second window while they follow along.{page}

SOURCE - the ONLY source of facts. Never use anything you remember about this project:
\"\"\"{source}\"\"\"

The video's script, for wording and ordering only:
\"\"\"{narration}\"\"\"

Write a guide someone can follow start to finish WITHOUT the video:
- A "# " title naming the project and what the guide gets done.
- One short paragraph, at most two sentences, saying what the project is.
- "## Requirements" - the hardware, OS, language and library versions, accounts, tokens or system
  packages the source names.
- "## Install" - the commands in order, in ``` fenced blocks, spelled exactly as the source spells
  them.
- "## First run" - the quickstart: the command that starts it, the flags that matter, and what the
  user should see, including any address or port.
- "## Settings worth knowing" - the flags, environment variables and options the source or the
  script calls out, each with one line on what it does.
- "## If it does not work" - three to six concrete problems and their fixes, taken from the source.

RULES:
- Every command, flag, path, package name, URL, port and version number is COPIED FROM THE SOURCE.
  If the source does not give one, leave it out. Never guess a command, a version or a package name.
- Anything the script spells out appears here too, in the script's own wording and order, and the
  guide contradicts the video nowhere.
- It works WITHOUT the video: no "as I mentioned", no timestamps, no "see above".
- Plain Markdown, self-contained: one "# " title line, short "## " sections, bullet lists, numbered
  steps, "- [ ] " checkboxes where something is worth ticking off, and ``` fenced blocks around
  everything meant to be copied verbatim. No HTML, no images, no tables - it is also read as plain
  text and printed as a PDF.
- At most {words} words and {max_chars} characters. Dense and usable: no introduction about the
  video, no filler, no sign-off. Never mention that the video is AI-generated.

Return JSON with keys:
- "title": the guide's own title, max 10 words, naming the project and what it gets done.
- "guide": the complete Markdown of the guide."""
    out = _chat([{"role": "system", "content": _system(channel, used=used, rules=False)},
                 {"role": "user", "content": user}])
    body = str(out.get("guide") or "").strip()
    if len(body) < 200:          # an empty or one-line answer is not a guide
        return {}
    name = re.sub(r"\s+", " ", str(out.get("title") or "")).strip().strip(chr(34))[:120]
    return {"kind": "setup guide", "title": name or f"The setup guide for {title}"[:120],
            "markdown": body[:max_chars]}


_HASHTAG = re.compile(r"(?<!\w)#(\w+)")


def _lowercase_hashtags(text: str) -> str:
    """YouTube hashtags are case-insensitive but display as written; keep them uniformly lowercase."""
    return _HASHTAG.sub(lambda m: "#" + m.group(1).lower(), text)


# Extra composition rules for a kids thumbnail. A toddler channel is browsed on a phone by an
# adult scrolling fast, and the thumbnails that win there are not the ones with the most in them:
# one huge face, one object, one flat loud colour. Anything else reads as noise at 168x94 px.
_THUMB_KIDS_RULE = (
    "  Compose it for a TODDLER audience scrolling on a phone: ONE character's face and upper body\n"
    "  filling most of the frame, eyes wide and mouth open in delight, plus at most ONE other\n"
    "  object, large and unmistakable. A flat, bright, high-contrast background (one strong colour\n"
    "  or a simple sky) - no scenery, no crowd, no small details, nothing in the distance.\n")


def write_thumbnail(title: str, narration: str, style: str, channel: dict | None = None,
                    used: list | None = None, idea: str = "") -> dict:
    """A single-image prompt for an eye-catching YouTube thumbnail.

    idea: the thumbnail the weekly plan asked for (services/planner.py); the prompt then realises
    that composition instead of inventing one. Empty for every other job."""
    from services import thumbtext   # lazy: thumbtext pulls storyboard, which imports this module

    ch = _channel(channel)
    kids = ch["audience"] == "kids"
    max_words, max_chars = thumbtext.caps(ch)
    emotion = "big expressive joyful emotion" if kids else "strong expressive emotion"
    wish = (f'\nThe thumbnail must show this, as planned for the video: "{idea.strip()}"'
            if idea.strip() else "")
    featured = cast.names(cast.main_characters(channel)) if used is None else list(used)
    subject = "the main character(s) and the key objects of the story"
    if not cast.enabled(channel):
        emotion = "strong, clear visual energy"
        subject = ("the topic itself - the screen, tool, object or moment the video is about, big and "
                   "unmistakable; an unnamed person (hands on a keyboard, someone at a laptop, a "
                   "presenter) may share the frame but is never named and never the point")
    elif featured:
        subject = (f"the video's cast character(s) {', '.join(featured)} - named and "
                   "described exactly as in the cast sheet - with the key objects of the story")
    user = f"""We produced a short video titled "{title}".
Its story: \"\"\"{narration[:1500]}\"\"\"
Its visual style: {style}{wish}

Design ONE eye-catching YouTube thumbnail image for it. A viewer glancing for half a second
must INSTANTLY understand what the video is about. Return JSON with key:
- "thumbnail_prompt": a prompt for an image generator describing a single dramatic scene that
  SHOWS THE VIDEO'S SUBJECT: {subject}, front and
  center, caught mid-action with {emotion}. Bold composition, exaggerated
  scale, vivid saturated colors, strong contrast and lighting, simple readable background.
{_THUMB_KIDS_RULE if kids else ""}  Keep the TOP-RIGHT corner free of important elements (a channel logo is placed there).
  Same world/style as the video.
  It must end with: "No text, subtitles, captions, letters as typography, logos or watermarks."
- "thumbnail_texts": 2-3 alternative overlay lines for this same picture, each at most
  {max_words} words and under {max_chars} characters, no full stops, no quotation marks. Each must be
  readable at thumbnail size on a phone and must say something the picture alone cannot - the
  stake, the number, the surprise. Best first. They are drawn over the picture afterwards, so the
  picture itself still contains no letters of any kind.
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, used=used)}, {"role": "user", "content": user}])
    tp = str(out.get("thumbnail_prompt", "")).strip()
    if not tp:
        raise RuntimeError("vLLM returned no thumbnail prompt")
    # Missing overlay lines is not an error: the video simply ships the clean render, which is
    # exactly what every video did before this existed.
    raw = out.get("thumbnail_texts") or out.get("thumbnail_text") or []
    if isinstance(raw, str):
        raw = [raw]
    texts, seen = [], set()
    for t in list(raw)[:4]:
        t = thumbtext.normalize(t, max_words, max_chars)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            texts.append(t)
    return {"thumbnail_prompt": tp, "thumbnail_texts": texts}


MUSIC_CAPTION_EXAMPLE = ("Global Metadata: Lo-fi hip-hop, chillhop. 78 BPM, D flat major. Laid-back and dreamy "
                         "throughout. Bedroom production: warm texture, vinyl crackle, soft-clipped drums.\\n\\n"
                         "Vocal Details: Soft androgynous vocal, hushed delivery, long stretches with no vocals.\\n\\n"
                         "Arrangement: Dusty boom-bap drums, low round sub bass, warm Rhodes piano chords. "
                         "Intro: solo chords fading in. Outro: elements fade one by one.")


SONG_VOICE_DESCRIPTIONS = {
    "auto": "",
    "child": "a single young child's voice (about 6-8 years old): sweet, bright, clear and innocent, gently sung",
    "girl": "a single young girl's voice: light, sweet and clear, gently sung",
    "boy": "a single young boy's voice: bright, clear and cheerful, gently sung",
    "female": "one warm, friendly female singer with clear, intelligible delivery",
    "male": "one warm, friendly male singer with clear, intelligible delivery",
}


def resolve_song_voice(voice, kids: bool) -> str:
    """The singer to ask for: an explicit choice wins; "auto" means a child for kids channels."""
    if voice in SONG_VOICE_DESCRIPTIONS and voice != "auto":
        return voice
    return "child" if kids else "auto"


_BPM_RE = re.compile(r"\b\d{2,3}(?:\s*(?:-|–|to)\s*\d{2,3})?\s*BPM\b", re.I)


def finalize_caption(caption: str, bpm: int | None = None, voice_desc: str | None = None) -> str:
    """Deterministic fixes to the LLM's music caption: the literal "\\n" copied from the example
    becomes a real newline, any tempo is replaced by the forced BPM (or one is inserted), and
    the requested voice description is guaranteed to sit in the Vocal Details section."""
    cap = (caption or "").replace("\\n", "\n").strip()
    if bpm:
        if _BPM_RE.search(cap):
            cap = _BPM_RE.sub(f"{bpm} BPM", cap)
        elif re.search(r"Global Metadata:\s*", cap, re.I):
            cap = re.sub(r"(Global Metadata:\s*)", lambda m: f"{m.group(1)}{bpm} BPM. ", cap, count=1, flags=re.I)
        else:
            cap = f"Global Metadata: {bpm} BPM.\n\n{cap}"
    if voice_desc:
        desc = voice_desc[0].upper() + voice_desc[1:]
        key = re.split(r"[:(,]", voice_desc)[0].strip().lower()   # e.g. "a single young child's voice"
        if key not in cap.lower():
            m = re.search(r"Vocal Details:\s*", cap, re.I)
            cap = (cap[:m.end()] + desc + ". " + cap[m.end():]) if m else f"{cap}\n\nVocal Details: {desc}."
    return cap


def needs_slowdown(measured: float, budget: float) -> bool:
    """A song sung clearly faster than its words-per-second budget gets one slower re-render."""
    return bool(measured) and measured > budget * config.SONG_PACE_TOLERANCE


def retry_reason(first: dict, budget: float, seconds_range: tuple | None) -> str:
    """Why a first song render earns its one re-render: "fast" (sung well over its
    words-per-second budget), "short" (the recording ended before the job's target length
    range) or "" (keep it). Pace wins when both are true - there is only one retry, and a
    raced-through song is usually short because it raced. A song LONGER than the range is
    never re-rendered: the recording is already hard-cut at max_seconds and the clip count
    simply follows the measured track."""
    if needs_slowdown(float(first.get("wps") or 0.0), budget):
        return "fast"
    lo = float(seconds_range[0]) if seconds_range else 0.0
    dur = float(first.get("duration") or 0.0)
    if lo and dur and dur < lo * config.SONG_SHORT_TOLERANCE:
        return "short"
    return ""


def write_music(title: str, narration: str, transcript: str,
                standalone: bool = False, seconds: float | None = None,
                max_seconds: float | None = None,
                seconds_range: tuple | None = None,
                channel: dict | None = None, used: list | None = None,
                song_voice: str = "auto", slow_down: dict | None = None,
                lengthen: dict | None = None, given_lyrics: str = "") -> dict:
    """Caption + lyrics for the MiniMax Music3 model, derived from the video content.

    standalone=True: the song IS the only audio of the video - the lyrics must tell the
    video's story in order, so the words being sung at any moment match the visuals.
    song_voice: config.SONG_VOICES entry ("auto" = a child for kids channels).
    slow_down: {"measured", "budget", "previous_lyrics"} when a first render was sung too
    fast - the lyrics are rewritten shorter and the pace instructions are stressed.
    lengthen: {"measured", "low", "target", "previous_lyrics"} when a first render ended
    before the job's target length range - the lyrics are rewritten longer at the same
    pace. Never passed together with slow_down.
    given_lyrics: the user's own lyrics (kids channels). They are sung VERBATIM: the LLM only
    writes the caption and adds section tags, and a deterministic guard puts the user's words
    back if it changed any. The returned dict then carries "lyrics_kept"."""
    transcript = transcript[:8000]
    given = (given_lyrics or "").strip() if standalone else ""   # the user's words: never rewritten
    given_words = len(clean_lyrics(given).split()) if given else 0
    kids = _channel(channel)["audience"] == "kids"
    voice = resolve_song_voice(song_voice, kids)
    voice_desc = SONG_VOICE_DESCRIPTIONS[voice]
    bpm = config.KIDS_SONG_BPM if kids else None
    # sung words per second: young children need slow, clearly enunciated singing
    wps = config.KIDS_SONG_WPS if kids else 2
    kids_rule = ("" if not kids else
                 " The audience is YOUNG CHILDREN (early education): the music must be soft, gentle and "
                 f"calm - nursery-rhyme pace at EXACTLY {bpm} BPM, simple sparse instrumentation (soft piano, "
                 "ukulele, glockenspiel, light hand percussion, warm pads), low dynamics, no busy or loud "
                 "arrangements, no dense drums, no fast rhythms. "
                 + (f"Vocals: slow, clear sing-along delivery - one sung word per beat, a breath-length pause "
                    f"between lines, about {wps:g} sung words per second, "
                    + ("" if given else "short simple lines of 4-6 words, ")     # the lines are the user's
                    + "a simple repeating melody, notes held long at the end of each line, so every word is "
                    "understood; keep instrumental gaps short. " if standalone else
                    "It sits far in the background so every narrated word stays clear. ")
                 + f"State the tempo ({bpm} BPM), the soft instruments and the gentle mood explicitly in the caption.")
    voice_rule = (f' The Vocal Details section of the caption MUST describe the singer as: "{voice_desc}".'
                  if standalone and voice_desc else "")
    slow_rule = ""
    if standalone and slow_down and given:
        slow_rule = (f"\n\nIMPORTANT: a previous render of these exact lyrics was sung TOO FAST - "
                     f"{float(slow_down.get('measured') or 0):.1f} words per second where the limit is "
                     f"{float(slow_down.get('budget') or wps):.1f}. The WORDS CANNOT CHANGE. Slow the MUSIC "
                     "down instead: a slower tempo, longer notes held at the end of every line, a clear "
                     "breath-length pause between lines, and say in the caption that every word is "
                     "enunciated slowly and unhurried.")
    elif standalone and slow_down:
        prev = str(slow_down.get("previous_lyrics") or "")
        prev_words = len(clean_lyrics(prev).split())
        slow_rule = (f"\n\nIMPORTANT: a previous version of this song was sung TOO FAST - "
                     f"{float(slow_down.get('measured') or 0):.1f} words per second where the limit is "
                     f"{float(slow_down.get('budget') or wps):.1f}. Rewrite the lyrics with FEWER words (at most "
                     f"{max(12, int(prev_words * 0.75))} words in total), shorter lines, longer held notes and clear "
                     "pauses between lines, keeping the same story order and ending. Previous lyrics:\n"
                     f'"""{prev}"""')
    long_rule = ""
    if standalone and lengthen and not slow_down and not given:
        prev = str(lengthen.get("previous_lyrics") or "")
        prev_words = len(clean_lyrics(prev).split())
        measured = float(lengthen.get("measured") or 0)
        low = float(lengthen.get("low") or 0)
        grow = min(2.5, max(1.3, float(lengthen.get("target") or low) / max(1.0, measured)))
        long_rule = (f"\n\nIMPORTANT: a previous version of this song came out TOO SHORT - "
                     f"{measured:.0f} seconds where the song must last at least {low:.0f} seconds. "
                     f"Rewrite the lyrics LONGER (at least {max(24, int(prev_words * grow))} words in total), "
                     "keeping the same story, the same order and the same ending: give every story beat its "
                     "own lines and ADD sections ([Verse 2], [Bridge], a repeated [Chorus]). Do NOT sing the "
                     "existing words faster - the pace must stay exactly as unhurried - and do not pad with "
                     "long instrumental sections. Previous lyrics:\n"
                     f'"""{prev}"""')
    if standalone:
        ceiling = int(max_seconds or ((seconds or 90) * 2))
        if given:
            length_rule = (f"The lyrics are FIXED: {given_words} sung words, about "
                           f"{int(seconds or given_words / max(0.1, wps))} seconds of singing at {wps:g} words "
                           "per second. Do NOT add or remove words to reach a length - shape the ARRANGEMENT "
                           "so every one of them is sung unhurried and the song reaches its END well before "
                           f"the {ceiling}-second hard cut.")
        elif seconds_range:
            lo, hi = int(seconds_range[0]), int(seconds_range[1])
            target = (lo + hi) // 2
            length_rule = (f"Aim for a song BETWEEN {lo} and {hi} seconds; the recording is HARD-CUT at "
                           f"{ceiling} seconds, so the lyrics MUST reach the story's ENDING and the song "
                           f"must finish within that range. Budget roughly {wps:g} sung words per second: about "
                           f"{int(target * wps)} words of lyrics in total across all sections. If the content "
                           f"cannot fit, compress or summarize it - never leave the story unfinished.")
        elif seconds:
            target = int(seconds)
            length_rule = (f"Aim for a song of about {target} seconds; the recording is HARD-CUT at "
                           f"{ceiling} seconds, so the lyrics MUST reach the story's ENDING and the song "
                           f"must finish well before that. Budget roughly {wps:g} sung words per second: about "
                           f"{int(target * wps)} words of lyrics in total across all sections. If the content "
                           f"cannot fit, compress or summarize it - never leave the story unfinished.")
        else:
            length_rule = (f"There is NO length target: give EVERY story beat its own lyric moment - do NOT "
                           f"compress or summarize the content (no skipping ahead). The recording is HARD-CUT "
                           f"at {ceiling} seconds, so the song must still reach its ENDING before that "
                           f"(budget roughly {wps:g} sung words per second when sizing the lyrics).")
        role = (("This track is the ONLY audio of the video - there is no narration. The lyrics are "
                 "ALREADY WRITTEN by the user and are FINAL: return them EXACTLY as given - same words, "
                 f"same order, nothing added, removed, reworded or translated. {length_rule} Your real "
                 "work is the CAPTION: pick the genre, tempo, key, mood and instrumentation that suit "
                 "THESE lyrics and this video. Prefer clear, intelligible vocals and keep "
                 "instrumental-only sections short.") if given else
                ("This track is the ONLY audio of the video - there is no narration. "
                 "The song must TELL THE VIDEO'S STORY IN ORDER: the lyrics follow the story beats "
                 "from beginning to end, so that whatever is being sung at any moment is what the "
                 f"video shows at that moment. {length_rule} Prefer clear, intelligible "
                 "vocals and keep instrumental-only sections short."))
    else:
        role = ("The track plays quietly UNDER a spoken narration as a background bed: it MUST be "
                "fully INSTRUMENTAL - no vocals, no singing, no lyrics, no spoken words. Choose "
                "gentle, soft, unobtrusive instrumentation with low intensity and smooth dynamics, "
                "and state clearly in the caption's Vocal Details that the track is instrumental "
                "with no vocals.")
    if given:
        lyrics_rule = ("the user's lyrics above, copied VERBATIM - not one sung word added, removed, "
                       "reordered or respelled. The ONLY edit allowed: if they carry no section tags, put "
                       "[Verse] / [Chorus] / [Bridge] / [Outro] tags on their own lines between the existing "
                       "lines (a pasted paragraph may also be broken into short sung lines). Never add an "
                       "intro line, an outro line, an ad-lib or a repeat.")
    elif standalone:
        lyrics_rule = ("song lyrics using section tags like [Intro], [Verse], [Chorus], "
                       "[Instrumental], [Outro]. "
                       + ("The lyrics are sung aloud: name any character, never describe how they "
                          "look." if cast.enabled(channel) else
                          "This channel has no characters: the lyrics carry the video's information, "
                          "sung to the viewer as 'you' - never invent a person, a name or a story."))
    else:
        lyrics_rule = 'exactly the string "[Instrumental]" - this background track has no words.'
    given_block = (f'\nThe FINAL lyrics, written by the user - return these words unchanged:\n'
                   f'\"\"\"{given}\"\"\"\n' if given else "")
    user = f"""We are scoring a short video titled "{title}".
Story of the video, in order:
\"\"\"{narration}\"\"\"

Original transcript excerpt (for flavor and vocabulary):
\"\"\"{transcript}\"\"\"
{given_block}
Design one music track for this video. {role}{kids_rule}{voice_rule}{slow_rule}{long_rule}
Return JSON with keys:
- "caption": a detailed music description in this exact three-part format (Global Metadata / Vocal Details / Arrangement), similar in structure to: "{MUSIC_CAPTION_EXAMPLE}" - but with genre, BPM, key, mood and instrumentation that FIT this video's content and tone.
- "lyrics": {lyrics_rule}
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system(channel, used=used, rules=False, spoken=True)},
                 {"role": "user", "content": user}])
    if not out.get("caption"):
        raise RuntimeError("vLLM returned no music caption")
    out.setdefault("lyrics", "")
    if given:
        # the words are the user's: an LLM that edited them (or returned none) loses its version
        out["lyrics_kept"] = "verbatim" if lyrics.same_words(out["lyrics"], given) else "verbatim-fallback"
        if out["lyrics_kept"] == "verbatim-fallback":
            out["lyrics"] = lyrics.ensure_sections(given)
    out["caption"] = finalize_caption(str(out["caption"]), bpm=bpm,
                                      voice_desc=voice_desc if standalone else None)
    out["song_voice"] = voice
    return out


def clean_lyrics(lyrics: str) -> str:
    """Strip section tags and parentheticals; return the sung words as flowing text."""
    text = re.sub(r"\[[^\]]*\]", " ", lyrics or "")
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    sample = sys.stdin.read() if not sys.stdin.isatty() else (
        "Today we look at how honeybees build their hives. Worker bees secrete wax and shape it into hexagonal "
        "cells, the most efficient shape for storing honey. The queen lays up to two thousand eggs per day.")
    plan = write_narration(sample, words=60)
    print(json.dumps(plan, indent=2))
    scenes = write_scenes(plan["narration"], plan["style"], 4)
    print(json.dumps(scenes, indent=2))
