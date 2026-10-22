#!/usr/bin/env python
"""How much like the uploaded portrait the presenter still looks, chunk by chunk.

A whole-video presenter on a chunked engine is a chain: each chunk is conditioned on the tail of
the one before it, so it continues the PICTURE of the last one and small deviations from the
portrait compound. Past roughly 30-40 s of one unbroken chain the face is recognisably someone
else. `AVATAR_CHAIN_MAX` breaks the chain every few chunks to bound that; this is how you see
whether it worked, without re-rendering anything.

    python scripts/avatar_identity.py output/<job id>
    python scripts/avatar_identity.py <portrait.png> <clip.mp4> [clip.mp4 ...]

The first form reads the job's own plan for which chunks were anchored and re-measures every
segment against the portrait it was rendered from, so it works on a job made before any of this
existed. The second measures any portrait and clips at all, which is what makes an ad-hoc pair
or an old spike directory usable.

What to read:
  - the TREND down a chain, not the absolute number. A move smaller than about 0.04 between
    chunks is inside the measurement's own error (see services.avatar.face_identity).
  - a jump back UP at an anchored chunk is the chain reset doing its job. That sawtooth is the
    evidence; a line that keeps falling through the anchors means they are not taking effect.
  - `faces` short of `frames` is the strongest signal there is: a likeness that has melted stops
    being detected as a face at all.

Needs the two CPU models in models/ (see docs/DEPLOYMENT.md section 2.1). CPU only - it never
touches the render box.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                                        # noqa: E402
from services import avatar                                          # noqa: E402


def _rows_from_job(d: Path):
    """(label, clip, portrait, anchored) per segment of a finished job."""
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    av = plan.get("avatar") or {}
    segs = av.get("segments") or []
    if not segs:
        sys.exit(f"{d}: no presenter segments on the plan")
    # the squared portrait the engine was actually handed, not the channel's stored upload
    port = d / "avatar_ref.png"
    if not port.exists():
        sys.exit(f"{d}: {port.name} is gone, so there is nothing to measure against")
    print(f"{d.name}: {len(segs)} segment(s) on {av.get('engine', '?')}, "
          f"coverage {av.get('coverage', '?')}, chain_max {av.get('chain_max', 'n/a')}")
    for i, s in enumerate(segs, 1):
        yield f"{i:>2} {s.get('start', 0):>6.2f}s", d / s["file"], port, bool(s.get("anchored"))


def _rows_from_args(port: Path, clips):
    for c in clips:
        yield Path(c).name[:22], Path(c), port, False


def main(argv):
    if not argv:
        sys.exit(__doc__)
    first = Path(argv[0])
    if first.is_dir():
        rows = list(_rows_from_job(first))
    elif len(argv) >= 2:
        rows = list(_rows_from_args(first, argv[1:]))
    else:
        sys.exit("give a job directory, or a portrait and at least one clip")

    if not avatar.face_models_ready():
        sys.exit(f"the face models are not in {config.MODELS_DIR} "
                 f"({config.AVATAR_FACE_DETECT_MODEL}, {config.AVATAR_FACE_EMBED_MODEL}) - "
                 f"see docs/DEPLOYMENT.md section 2.1")

    print(f"\n{'chunk':<22} {'anchor':>6} {'mean':>7} {'min':>7} {'faces':>8}   drift")
    print("-" * 72)
    prev, first_mean, means = None, None, []
    for label, clip, port, anchored in rows:
        if not clip.exists():
            print(f"{label:<22} {'-':>6} {'missing':>7}")
            continue
        got = avatar.face_identity(clip, port)
        if got is None:
            print(f"{label:<22} {'yes' if anchored else '':>6} {'unreadable':>7}")
            continue
        means.append(got["mean"])
        first_mean = got["mean"] if first_mean is None else first_mean
        step = "" if prev is None else f"{got['mean'] - prev:+.3f}"
        flag = ""
        if got["faces"] < got["frames"]:
            flag = "  NO FACE in some frames"
        if got["min"] < config.AVATAR_FACE_MIN_COSINE:
            flag += "  BELOW SAME-PERSON THRESHOLD"
        print(f"{label:<22} {'yes' if anchored else '':>6} {got['mean']:>7.3f} "
              f"{got['min']:>7.3f} {got['faces']:>4}/{got['frames']:<3} {step:>7}{flag}")
        prev = got["mean"]

    if len(means) > 1:
        print("-" * 72)
        print(f"first {means[0]:.3f} -> last {means[-1]:.3f} "
              f"({means[-1] - means[0]:+.3f} over {len(means)} chunks), "
              f"worst chunk {min(means):.3f}")
        if abs(means[-1] - means[0]) < 0.04:
            print("that is inside the measurement's own error - no drift this run")


if __name__ == "__main__":
    main(sys.argv[1:])
