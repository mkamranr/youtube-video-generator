"""vLLM (OpenAI-compatible) planning: narration first, then one scene prompt per narration segment.

Two-step design: the narration is written and voiced first; only then — once the real clip count
is known from the audio duration — are the scene prompts generated, one per narration segment,
so the visuals stay in sync with what is being said all the way to the end.
"""
import json
import re
import sys

from openai import OpenAI

import config

def _system() -> str:
    return ("You are a creative director turning a YouTube video transcript into a short, original "
            "AI-generated animated video FOR CHILDREN. All content must be family-friendly, gentle "
            "and safe for young kids: no violence, gore, fear, romance, innuendo or mature themes - "
            "even dark or spooky styles stay playful and reassuring. "
            f'The videos are published on the kids YouTube channel "{config.CHANNEL_NAME}": if the '
            "source content mentions any other channel, brand, show or creator name, NEVER use it - "
            f'use "{config.CHANNEL_NAME}" instead wherever a name is called for, and mention the '
            "channel naturally only where it fits (intros, outros, sign-offs), without forcing it "
            "into every line. "
            "Always answer with a single JSON object and nothing else.")

STYLE_EXAMPLE = ("Realistic live-action cinematic look, practical film photography style, anamorphic lens, "
                 "shallow depth of field, film grain, restrained grading for a premium feel, natural movement.")

SCENE_RULES = """Each scene prompt describes ONE continuous {secs}-second shot for a text-to-video model and must contain, in this order:
- Scene overview: subject, setting, action, mood (2-3 sentences).
- Storyboard with timestamps, e.g. "[0s-2s] ... [2s-4s] ... [4s-{secs}s] ...".
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
    return OpenAI(base_url=config.VLLM_URL, api_key="none")


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
        r = c.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        r = c.chat.completions.create(**kwargs)
    msg = r.choices[0].message
    content = msg.content or getattr(msg, "reasoning", None) or ""
    if not content.strip():
        raise RuntimeError(f"vLLM returned no content (finish_reason={r.choices[0].finish_reason})")
    return _parse_json(content)


def _as_strings(scenes) -> list:
    return [s if isinstance(s, str) else s.get("prompt", json.dumps(s)) for s in scenes]


def write_narration(transcript: str, words: int | None = None, art_style: str = "") -> dict:
    """Step 1: title, narration script and global visual style (no scenes yet)."""
    words = words or config.NARRATION_WORDS
    transcript = transcript[: config.MAX_TRANSCRIPT_CHARS]
    if art_style:
        style_rule = (f'one sentence describing the global visual style applied to every shot. '
                      f'It MUST follow this art direction: "{art_style}" - keep that exact aesthetic '
                      f'and expand it with detail that fits this video\'s subject.')
    else:
        style_rule = f'one sentence describing the global visual style applied to every shot, similar to: "{STYLE_EXAMPLE}"'
    user = f"""Transcript of the source video:
\"\"\"{transcript}\"\"\"

Create a new short video inspired by this content. Return JSON with keys:
- "title": short title.
- "narration": a voiceover script of about {words} words, plain spoken English, no headings, no stage directions, no emojis. It will be read aloud by a TTS voice. Stay close to {words} words.
- "style": {style_rule}
"""
    plan = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}])
    for key in ("title", "narration", "style"):
        plan.setdefault(key, "")
    if not plan["narration"].strip():
        raise RuntimeError("vLLM returned an empty narration")
    if config.SCRIPT_REVIEW:
        plan["narration"], plan["review_issues"] = review_script(
            plan["narration"], f'a video titled "{plan.get("title", "")}"')
    return plan


def write_script_from_prompt(text: str, source_kind: str, art_style: str = "",
                             words: int | None = None, feedback: str = "",
                             previous: str | None = None) -> dict:
    """Draft a video script (title/narration/style) from a user topic or description."""
    words = words or config.NARRATION_WORDS
    if art_style:
        style_rule = (f'one sentence describing the global visual style applied to every shot. '
                      f'It MUST follow this art direction: "{art_style}" - keep that exact aesthetic '
                      f'and expand it with detail that fits this video.')
    else:
        style_rule = f'one sentence describing the global visual style applied to every shot, similar to: "{STYLE_EXAMPLE}"'
    if source_kind == "description":
        task = ('The user wrote the following description / short script for a video. Build the video '
                'script from it, staying FAITHFUL to their content, characters, structure and intent - '
                'expand and polish it, do not replace it:\n"""' + text + '"""')
    else:
        task = 'Create an original kids video script about this topic: "' + text + '"'
    revision = ""
    if feedback:
        revision = '\n\nThe user reviewed a previous draft and asked for these changes: "' + feedback + '"'
        if previous:
            revision += '\nPrevious draft narration:\n"""' + previous + '"""\nRevise it accordingly.'
    user = f"""{task}{revision}

Return JSON with keys:
- "title": short title.
- "narration": the full script as a voiceover of about {words} words, plain spoken English, no headings, no stage directions, no emojis. It will drive the whole video. Stay close to {words} words.
- "style": {style_rule}
"""
    plan = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}])
    for key in ("title", "narration", "style"):
        plan.setdefault(key, "")
    if not plan["narration"].strip():
        raise RuntimeError("vLLM returned an empty script")
    if config.SCRIPT_REVIEW:
        plan["narration"], plan["review_issues"] = review_script(
            plan["narration"], f'a video about: "{text[:300]}"')
    return plan


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


def write_scenes(narration: str, style: str, n: int, on_progress=None,
                 kind: str = "voiceover narration", segments: list | None = None,
                 on_review_issue=None) -> list:
    """Step 2: exactly n scene prompts, generated ONE PER CALL so scene i cannot drift
    away from narration segment i (a single big call loses sync on long lists).

    segments: optional pre-computed per-clip text (e.g. whisper-timed lyrics); empty
    entries mean nothing is heard during that clip (instrumental/silence)."""
    if segments is None:
        segments = split_narration(narration, n)
    segments = (list(segments) + [""] * n)[:n]
    scenes = []
    for i, seg in enumerate(segments):
        if on_progress:
            on_progress(f"scene {i + 1}/{n}")
        prev = (f'The previous clip\'s scene was:\n"{scenes[-1]}"\n'
                "MANDATORY: this clip's [0s] beat continues EXACTLY from that scene's ending frame - same "
                "camera angle, same characters in the same positions, same setting and lighting. Do not cut "
                "to a new shot or angle at the start; evolve the action from that frozen moment.") if scenes \
            else "This is the FIRST clip of the video: establish the world."
        nxt = (f'For flow only, the next segment will be: "{segments[i + 1]}"') if i + 1 < n \
            else "This is the FINAL clip: give it a conclusive ending."
        if seg.strip():
            heard = f"""While this clip plays, the {kind} is EXACTLY this:
"{seg}"

Write ONE scene prompt for this clip that visualizes exactly what is heard during it.
EVERY object, creature, letter or concept mentioned above MUST appear in the storyboard —
give each its own moment within the {config.CLIP_SECONDS} seconds. Do not show things from other segments
except as a continuation of the previous shot."""
        else:
            heard = f"""During this clip the {kind} is INSTRUMENTAL - no words are heard.
Write ONE scene prompt that carries the story visually: continue the previous shot's motion,
build atmosphere, or transition toward the next segment. Introduce nothing that belongs to
other segments' content."""
        user = f"""Global visual style: {style}

We are storyboarding clip {i + 1} of {n} for a video. Each clip is {config.CLIP_SECONDS} seconds.
The soundtrack is a {kind}.
{prev}
{nxt}
{heard}
{SCENE_RULES.format(secs=config.CLIP_SECONDS)}
Return JSON: {{"scene": "the scene prompt string"}}"""
        out = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}])
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
                                         prev_scene=scenes[-1] if scenes else "")
            if issues and on_review_issue:
                on_review_issue(i + 1, issues)
        scenes.append(scene)
    return scenes


def review_scene(scene: str, segment: str, style: str, i: int, n: int,
                 kind: str = "voiceover narration", prev_scene: str = "") -> tuple:
    """Second-pass review of one scene prompt against the audio it must match.

    Returns (corrected_scene, issues). Catches inconsistencies like a letter or object
    in the storyboard not matching what the audio actually says at that moment."""
    if segment.strip():
        audio_ctx = f'While this clip plays, the {kind} is EXACTLY this:\n"{segment}"'
        rule = ("Every letter, word, object, creature, name and count that the storyboard displays or "
                "features MUST match what the audio says during this clip - e.g. if the audio says "
                "\"F is fairy\", the scene must show the letter F with the fairy, never a different letter. "
                "IMPORTANT: the audio text comes from automatic speech transcription and may contain small "
                "mis-hearings - if it pairs a letter with a word that clearly starts with a DIFFERENT letter "
                "(e.g. \"A is for fairy\"), trust the word: fairy belongs to the letter F, so show F.")
    else:
        audio_ctx = f"During this clip the {kind} is INSTRUMENTAL - no words are heard."
        rule = "The scene must not introduce letters/objects that belong to other clips' audio."
    continuity = ""
    if prev_scene.strip():
        continuity = f"""5. SEAMLESS CONTINUITY: this clip is rendered starting from the LAST frame of the previous clip.
The previous clip's scene was:
\"\"\"{prev_scene}\"\"\"
This scene's [0s] beat must directly continue that scene's ending (same camera angle, same characters in
the same positions, same setting); it must NOT open with a cut, a new shot, a different angle or location.
Reframing may only happen through continuous camera motion within the clip."""
    user = f"""You are reviewing a scene prompt for clip {i} of {n} of a children's video before it is rendered.
Global visual style: {style}

{audio_ctx}

Scene prompt to review:
\"\"\"{scene}\"\"\"

Check strictly:
1. {rule}
2. Internal consistency (no contradictions within the storyboard).
3. Content is gentle and safe for young children.
4. The required format is kept (overview, timestamped storyboard, camera, audio, the closing no-text line).
{continuity}

If everything is correct, return the scene UNCHANGED. If there are issues, FIX them with the smallest
possible edits, keeping everything else identical.
Return JSON: {{"issues": ["short description of each issue found, empty list if none"], "scene": "the final scene prompt"}}"""
    out = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}],
                max_tokens=4000)
    fixed = str(out.get("scene") or "").strip()
    issues = [str(x) for x in (out.get("issues") or []) if str(x).strip()]
    if not fixed:
        return scene, []
    return fixed, issues


def review_script(narration: str, context: str = "") -> tuple:
    """One review pass over a drafted script. Returns (corrected_narration, issues)."""
    ctx = f"\nContext: {context}" if context else ""
    user = f"""You are reviewing the final voiceover script of a children's video before production.{ctx}

Script:
\"\"\"{narration}\"\"\"

Check strictly: factual slips, internal inconsistencies (e.g. a letter or item paired with the wrong
word), ordering mistakes, anything not gentle and kid-safe, awkward phrasing that would sound wrong
read aloud. If everything is correct, return the script UNCHANGED; otherwise fix it with minimal edits.
Return JSON: {{"issues": ["each issue found, empty list if none"], "narration": "the final script"}}"""
    out = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}],
                max_tokens=4000)
    fixed = str(out.get("narration") or "").strip()
    issues = [str(x) for x in (out.get("issues") or []) if str(x).strip()]
    if not fixed:
        return narration, []
    return fixed, issues


def write_seo(title: str, narration: str, transcript: str) -> dict:
    """YouTube-ready SEO title and description for the generated video."""
    user = f"""We produced a short AI-generated video titled "{title}".
What the video says (its narration/story):
\"\"\"{narration}\"\"\"

Original source transcript excerpt (for topic keywords):
\"\"\"{transcript[:4000]}\"\"\"

Write YouTube upload metadata optimized for SEO and reach. Return JSON with keys:
- "seo_title": one compelling, keyword-rich title, max 70 characters, no clickbait lies, no quotes, no emojis at the start (one emoji at the end is allowed if it fits the topic).
- "seo_description": a description of 2-3 short paragraphs (120-200 words total): the first sentence hooks and contains the main keyword; naturally weave in related search keywords; end with a call-to-action inviting viewers to subscribe to "{config.CHANNEL_NAME}" and then 4-6 relevant hashtags on the last line.
Never mention that the video is AI-generated unless the content is about AI. Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}])
    return {"seo_title": str(out.get("seo_title", "")).strip()[:100],
            "seo_description": str(out.get("seo_description", "")).strip()}


def write_thumbnail(title: str, narration: str, style: str) -> dict:
    """A single-image prompt for an eye-catching YouTube thumbnail."""
    user = f"""We produced a short video titled "{title}".
Its story: \"\"\"{narration[:1500]}\"\"\"
Its visual style: {style}

Design ONE eye-catching YouTube thumbnail image for it. A viewer glancing for half a second
must INSTANTLY understand what the video is about. Return JSON with key:
- "thumbnail_prompt": a prompt for an image generator describing a single dramatic scene that
  SHOWS THE VIDEO'S SUBJECT: the main character(s) and the key objects of the story, front and
  center, caught mid-action with big expressive joyful emotion. Bold composition, exaggerated
  scale, vivid saturated colors, strong contrast and lighting, simple readable background.
  Keep the TOP-RIGHT corner free of important elements (a channel logo is placed there).
  Same world/style as the video.
  It must end with: "No text, subtitles, captions, letters as typography, logos or watermarks."
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}])
    tp = str(out.get("thumbnail_prompt", "")).strip()
    if not tp:
        raise RuntimeError("vLLM returned no thumbnail prompt")
    return {"thumbnail_prompt": tp}


MUSIC_CAPTION_EXAMPLE = ("Global Metadata: Lo-fi hip-hop, chillhop. 78 BPM, D flat major. Laid-back and dreamy "
                         "throughout. Bedroom production: warm texture, vinyl crackle, soft-clipped drums.\\n\\n"
                         "Vocal Details: Soft androgynous vocal, hushed delivery, long stretches with no vocals.\\n\\n"
                         "Arrangement: Dusty boom-bap drums, low round sub bass, warm Rhodes piano chords. "
                         "Intro: solo chords fading in. Outro: elements fade one by one.")


def write_music(title: str, narration: str, transcript: str,
                standalone: bool = False, seconds: float | None = None,
                max_seconds: float | None = None,
                seconds_range: tuple | None = None) -> dict:
    """Caption + lyrics for the MiniMax Music3 model, derived from the video content.

    standalone=True: the song IS the only audio of the video - the lyrics must tell the
    video's story in order, so the words being sung at any moment match the visuals."""
    transcript = transcript[:8000]
    if standalone:
        ceiling = int(max_seconds or ((seconds or 90) * 2))
        if seconds_range:
            lo, hi = int(seconds_range[0]), int(seconds_range[1])
            target = (lo + hi) // 2
            length_rule = (f"Aim for a song BETWEEN {lo} and {hi} seconds; the recording is HARD-CUT at "
                           f"{ceiling} seconds, so the lyrics MUST reach the story's ENDING and the song "
                           f"must finish within that range. Budget roughly 2 sung words per second: about "
                           f"{target * 2} words of lyrics in total across all sections. If the content "
                           f"cannot fit, compress or summarize it - never leave the story unfinished.")
        elif seconds:
            target = int(seconds)
            length_rule = (f"Aim for a song of about {target} seconds; the recording is HARD-CUT at "
                           f"{ceiling} seconds, so the lyrics MUST reach the story's ENDING and the song "
                           f"must finish well before that. Budget roughly 2 sung words per second: about "
                           f"{target * 2} words of lyrics in total across all sections. If the content "
                           f"cannot fit, compress or summarize it - never leave the story unfinished.")
        else:
            length_rule = (f"There is NO length target: give EVERY story beat its own lyric moment - do NOT "
                           f"compress or summarize the content (no skipping ahead). The recording is HARD-CUT "
                           f"at {ceiling} seconds, so the song must still reach its ENDING before that "
                           f"(budget roughly 2 sung words per second when sizing the lyrics).")
        role = ("This track is the ONLY audio of the video - there is no narration. "
                "The song must TELL THE VIDEO'S STORY IN ORDER: the lyrics follow the story beats "
                "from beginning to end, so that whatever is being sung at any moment is what the "
                f"video shows at that moment. {length_rule} Prefer clear, intelligible "
                "vocals and keep instrumental-only sections short.")
    else:
        role = ("The track plays UNDER a spoken narration, so prefer sparse vocals or "
                "instrumental-leaning writing.")
    user = f"""We are scoring a short video titled "{title}".
Story of the video, in order:
\"\"\"{narration}\"\"\"

Original transcript excerpt (for flavor and vocabulary):
\"\"\"{transcript}\"\"\"

Design one music track for this video. {role}
Return JSON with keys:
- "caption": a detailed music description in this exact three-part format (Global Metadata / Vocal Details / Arrangement), similar in structure to: "{MUSIC_CAPTION_EXAMPLE}" - but with genre, BPM, key, mood and instrumentation that FIT this video's content and tone.
- "lyrics": song lyrics using section tags like [Intro], [Verse], [Chorus], [Instrumental], [Outro].
Never mention real people's names or brands."""
    out = _chat([{"role": "system", "content": _system()}, {"role": "user", "content": user}])
    if not out.get("caption"):
        raise RuntimeError("vLLM returned no music caption")
    out.setdefault("lyrics", "")
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
